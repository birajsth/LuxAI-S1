import wandb
import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from ..storage.data_buffer import DataBuffer
from ..algorithms.mappo import MAPPO as TrainAlgo
from ..arch.policy import PolicyN as Policy



class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        #self.envs = config['envs']
        self.device = config['device']
        self.wandb_run = config['wandb_run']
        self.luxagent = config['agent']

        # parameters
        self.env_name = self.all_args.env_name
        self.num_envs = self.all_args.num_envs
        self.total_timesteps = self.all_args.total_timesteps
        self.batch_size = self.all_args.batch_size

        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.exp_name

        self.num_env_steps = self.all_args.num_steps


        self.use_linear_lr_decay = self.all_args.anneal_lr
        self.use_popart = self.all_args.use_popart

        self.use_lstm = self.all_args.use_lstm

        self.use_wandb = self.all_args.use_wandb
        self.use_artifact = self.all_args.use_artifact

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.checkpoint_dir = self.all_args.checkpoint_dir
        self.save_checkpoints = self.all_args.save_checkpoints

        self.log_dir = config["log_dir"]
        self.save_dir = config["save_dir"]
        self.run_name = config["run_name"]
        
        #self.model_dir = self.save_dir

        self.writer = SummaryWriter(self.log_dir +"/"+ self.run_name)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.all_args).items()])),
        )
        
   
        # policy network
        self.policy = Policy(self.all_args).to(self.device)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        # train track
        self.update = 0
        self.global_step = 0
        
        # load model from wandb artifact
        if self.use_artifact:
            if "checkpoint" in self.use_artifact:
                checkpoint = self.wandb_run.use_artifact(self.use_artifact)
                self.checkpoint_dir = checkpoint.download()
                print("Loaded checkpoint from wandb artifact")
            else:
                model = self.wandb_run.use_artifact(self.use_artifact)
                self.model_dir = model.download()
                print("Loaded model from wandb artifact")
            
        if self.checkpoint_dir:
            self.restore_from_checkpoint()
        elif self.model_dir:
            self.restore()

        
        # buffers
        share_observation_space = None
        self.buffer = DataBuffer(self.all_args,
                                self.luxagent.get_obs_space(),
                                share_observation_space,
                                self.luxagent.get_action_space())
        

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_value(self.buffer.next_obs_scalar, self.buffer.next_obs_spatial, self.buffer.next_available_actions).flatten()
        self.buffer.compute_returns(next_values, self.next_ids, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")
        if self.use_wandb:
            artifact = wandb.Artifact(f"{self.experiment_name}_model", type='model')
            artifact.add_dir(self.save_dir)
            self.wandb_run.log_artifact(artifact)

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
  
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)
        if self.trainer._use_valuenorm:
            policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
            self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
        print("Model Restored")

    def save_checkpoint(self):
        """Save checkpoint"""
        torch.save({'global_step': self.global_step,
                    'update': self.update,
                    'actor_state_dict': self.trainer.policy.actor.state_dict(),
                    'critic_state_dict':self.trainer.policy.critic.state_dict(),
                    'actor_optim_state_dict': self.trainer.policy.actor_optimizer.state_dict(),
                    'critic_optim_state_dict':self.trainer.policy.critic_optimizer.state_dict(),
                    'vnorm_state_dict': self.trainer.value_normalizer.state_dict() if self.trainer._use_valuenorm else None
                    }, 
                    str(self.save_dir) + '/checkpoint.pth')
        if self.use_wandb:
            artifact = wandb.Artifact(f"{self.experiment_name}_checkpoint", type='checkpoint')
            artifact.add_file(str(self.save_dir) + '/checkpoint.pth')
            self.wandb_run.log_artifact(artifact)

    def restore_from_checkpoint(self):
        """Restore Checkpoint"""
        checkpoint = torch.load(str(self.checkpoint_dir) + '/checkpoint.pth')
        self.global_step = checkpoint['global_step']
        self.update = checkpoint['update']
        self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        if self.trainer._use_valuenorm:
            self.trainer.value_normalizer.load_state_dict(checkpoint['vnorm_state_dict'])
        print("Checkpoint Restored")
 
    def log_train(self, train_infos, global_step):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                self.wandb_run.log({k: v}, step=global_step)
            else:
                self.writer.add_scalars(k, {k: v}, global_step)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    self.wandb_run.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writer.add_scalars(k, {k: np.mean(v)}, total_num_steps)