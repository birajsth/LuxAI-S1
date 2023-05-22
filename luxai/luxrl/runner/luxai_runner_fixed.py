import time
import torch
import random
from functools import reduce

import gym
from stable_baselines3.common.vec_env import SubprocVecEnv

from ...luxenv.luxenv.lux_env import LuxEnvironment
from ...luxenv.game.constants import LuxMatchConfigs_Default
from ...luxenv.luxenv.agent import Agent
from ...luxenv.luxenv.player import LuxPlayer

from ..utils.utils import slice_array, merge_actions
from .base_runner_fixed import Runner




def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment 
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def thunk():
        env = gym.wrappers.RecordEpisodeStatistics(local_env)
        env.seed(seed + rank)
        return local_env

    return thunk

class LuxAIRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(LuxAIRunner, self).__init__(config)

        self.next_ids = []

    def run(self):
        self.env_setup()
        self.warmup()   

        start = time.time()
        
        # approximate estimate
        # self.batch_size is a rough estimate just for update purpose
        num_updates = self.total_timesteps // self.batch_size
        
        self.total_env_steps = 0
        self.env_step = 0

        steps = 0
        
        # number of wins and total episodes
        self.num_wins = 0
        self.total = 0
        while self.global_step < self.total_timesteps:
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(self.update, num_updates)
            
            for step in range(self.num_env_steps):
                self.total_env_steps += 1 * self.num_envs
                # Sample actions
                values, actions, action_log_probs, hidden_states, actions_env_player = self.collect()
                
                self.env_step += step
                steps += actions.shape[0]
                self.global_step += actions.shape[0]

                # opponent actions
                actions_env_opponent = [[] for _ in range(self.num_envs)]

                actions_env = merge_actions(actions_env_player, actions_env_opponent)
                # Player state, reward and done
                states, player_rewards, player_dones, infos = self.envs.step(actions_env)

                # Get agent reward, done from their respective player
                rewards, dones = self.get_reward_done_from_player(states, actions, player_rewards, player_dones, infos)
                
                # Get next_agent_obs from their respective player
                next_obs_scalar, next_obs_spatial, next_available_actions = self.get_obs_from_player(states)

                
                data_player = next_obs_scalar, next_obs_spatial, next_available_actions, \
                      rewards, dones, values, actions, action_log_probs, hidden_states

                # insert data into buffer
                self.insert(data_player)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # increment update
            self.update += 1
            
            # save model
            if (self.update % self.save_interval == 0 or self.update == num_updates - 1):
                if self.save_checkpoints:
                    self.save_checkpoint()
                else:
                    self.save()

            # log information
            if self.update % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} updates, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                self.update,
                                num_updates,
                                self.global_step,
                                self.total_timesteps,
                                int(steps / (end - start))))
                train_infos["charts/avg_reward"] = self.buffer.rewards.mean().item()
                train_infos["charts/avg_return"] = self.buffer.returns.mean().item()
                train_infos["charts/SPS"] = int(steps / (end - start))
                train_infos["charts/num_updates"] = self.update
                train_infos["charts/num_env_steps"] = self.total_env_steps
                train_infos["charts/learning_rate_actor"] = self.policy.actor_optimizer.param_groups[0]["lr"]
                train_infos["charts/learning_rate_critic"] = self.policy.critic_optimizer.param_groups[0]["lr"]
            
                train_infos["charts/avg_win_rates"] = self.num_wins / self.total
                self.log_train(train_infos, self.global_step)
            # eval
            if self.update % self.eval_interval == 0 and self.use_eval:
                self.eval(self.global_step)

            # clear data from buffer
            self.buffer.reset()

            

    def warmup(self):
        # reset env
        states = self.envs.reset()

        next_obs_scalar, next_obs_spatial, next_available_actions = self.get_obs_from_player(states)
        self.buffer.nextinsert(next_obs_scalar, next_obs_spatial, next_available_actions, self.next_ids)

    def env_setup(self):
        # Create a RL agent in training mode
        self.players = [LuxPlayer(mode="train", model=self.policy, team_sprit=0.) for _ in range(self.num_envs)]
        self.opponents = [Agent() for _ in range(self.num_envs)]
        self.envs = SubprocVecEnv([make_env(LuxEnvironment(configs=LuxMatchConfigs_Default,
                                                            learning_agent=self.players[i],
                                                            opponent_agent=self.opponents[i]), i) for i in range(self.num_envs)]
        )

    

    def get_obs_from_player(self, states):
        self.next_ids = []
        self.num_agents = [0] * self.buffer.num_envs
        next_obs_scalar = torch.empty((0,) + self.buffer.obs_shape["scalar"], dtype=torch.float32).to(self.device)
        next_obs_spatial = torch.empty((0,) + self.buffer.obs_shape["spatial"], dtype=torch.float32).to(self.device)
        next_available_actions = torch.empty((0,) + self.buffer.available_actions_shape, dtype=torch.float32).to(self.device)
        for i in range(self.num_envs):
            agents = self.players[i].get_agent_obs(states[i])
            while True:
                try:
                    (agent_id, agent_obs) = next(agents)
                    self.next_ids.append((i, self.buffer.episodes[i], agent_id))
                    self.num_agents[i] += 1
                    
                    next_obs_scalar = torch.cat((next_obs_scalar, torch.tensor(agent_obs["scalar"], dtype=torch.float32).to(self.device).unsqueeze(0)), dim=0)
                    next_obs_spatial = torch.cat((next_obs_spatial, torch.tensor(agent_obs["spatial"], dtype=torch.float32).to(self.device).unsqueeze(0)), dim=0)
                    next_available_actions = torch.cat((next_available_actions, torch.tensor(agent_obs["available_actions"], dtype=torch.float32).to(self.device).unsqueeze(0)), dim=0)                
                except StopIteration:
                    # No more agents
                    # increment env episode
                    break
        return next_obs_scalar, next_obs_spatial, next_available_actions
    
    


    def get_reward_done_from_player(self, states, actions, player_rewards, player_dones, infos):
         # Get agent reward, done from their respective player
        rewards = []
        dones = []
        i = 0
        for env_id, _, agent_id in self.next_ids:
            if player_dones[env_id]:
                env_state = infos[env_id]["terminal_state"]   
                done = True  
            else:
                env_state = states[env_id]  
            reward, done = self.players[env_id].get_reward_done(env_state, agent_id, int(actions[i][0]), player_rewards[env_id])
            rewards.append(reward)

            dones.append(done)
            i += 1
        for i in range(self.num_envs):
            if player_dones[i]:
                self.buffer.episodes[i] += 1
                # updata payoff win/loss state
                if infos[i]["result"]["learning_agent"] == "win":
                    self.num_wins += 1
                self.total += 1
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        return rewards, dones
    
    @torch.no_grad()
    def collect(self):
        self.trainer.prep_rollout()
        hidden_state = None
        if self.use_lstm:
            hidden_state = (self.buffer.next_lstm_states_hidden, self.buffer.next_lstm_states_cell)
        actions, action_log_probs, _, values, hidden_states = self.trainer.policy.get_action_and_value(self.buffer.next_obs_scalar, self.buffer.next_obs_spatial,\
                                                                                                   self.buffer.next_available_actions, hidden_state)
        # rearrange actions
        # list consisting of agents actions for the env 
        actions_env = slice_array(actions.detach().cpu().numpy(), self.num_agents)
        values = values.flatten()
        return values, actions, action_log_probs, hidden_states, actions_env
    
    
    def insert(self, data):
        next_obs_scalar, next_obs_spatial, next_available_actions, \
            rewards, dones, values, actions, action_log_probs, hidden_states = data
        self.buffer.insert(actions, action_log_probs, values, rewards, hidden_states, dones)
        self.buffer.nextinsert(next_obs_scalar, next_obs_spatial, next_available_actions, self.next_ids)
        

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_value(self.buffer.next_obs_scalar, self.buffer.next_obs_spatial, self.buffer.next_available_actions).flatten()
        self.buffer.compute_returns(next_values, self.next_ids, self.trainer.value_normalizer)


