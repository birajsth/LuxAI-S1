import numpy as np
import torch
import torch.nn as nn

from ..arch.libs.value_norm import ValueNorm


class MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.policy = policy

       
        self.ppo_epoch = args.update_epochs
        self.norm_adv = args.norm_adv
        self.max_minibatch_size = args.max_minibatch_size
        
     
        self.max_grad_norm = args.max_grad_norm
        self.clip_coef = args.clip_coef
        self.clip_vloss = args.clip_vloss
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.target_kl = args.target_kl
        self.norm_adv = args.norm_adv

        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        
        self.use_lstm = args.use_lstm

        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None


    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.
        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        mb_obs_scalar, mb_obs_spatial, mb_available_actions, mb_hidden_state, \
        mb_actions, mb_logprobs, mb_values, mb_returns, mb_advantages = sample


        _, newlogprob, entropy, newvalue, _ = self.policy.get_action_and_value(mb_obs_scalar, mb_obs_spatial, mb_available_actions,\
                                            mb_hidden_state, mb_actions )
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            self.clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]


        if self.norm_adv and mb_advantages.shape[0]>1:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(mb_returns)
            mb_returns = self.value_normalizer.normalize(mb_returns)

        if self.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
         
        # Actor update
        self.policy.actor_optimizer.zero_grad()
        (pg_loss - self.ent_coef * entropy_loss).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        # Critic update
        self.policy.critic_optimizer.zero_grad()
        (v_loss * self.vf_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        return v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, ratio, actor_grad_norm, critic_grad_norm

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.
        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}

        train_info['losses/value_loss'] = 0
        train_info['losses/policy_loss'] = 0
        train_info['losses/entropy_loss'] = 0
        train_info['losses/old_approx_kl'] = 0
        train_info['losses/approx_kl'] = 0
        train_info['losses/clipfrac'] = 0
        train_info['losses/ratio'] = 0
        train_info['charts/actor_grad_norm'] = 0
        train_info['charts/critic_grad_norm'] = 0
        
        self.clipfracs = []
        for _ in range(self.ppo_epoch):
            
            data_generator = buffer.feed_forward_generator(self.max_minibatch_size)

            for sample in data_generator:

                v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, ratio, \
                    actor_grad_norm, critic_grad_norm \
                    = self.ppo_update(sample)

                train_info['losses/value_loss'] += v_loss.item()
                train_info['losses/policy_loss'] += pg_loss.item()
                train_info['losses/entropy_loss'] += entropy_loss.item()
                train_info['losses/old_approx_kl'] += old_approx_kl.item()
                train_info['losses/approx_kl'] += approx_kl.item()
                train_info['losses/clipfrac'] += np.mean(self.clipfracs)
                train_info['losses/ratio'] += ratio.mean()
                train_info['charts/actor_grad_norm'] += actor_grad_norm
                train_info['charts/critic_grad_norm'] += critic_grad_norm
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        num_updates = self.ppo_epoch * buffer.num_mini_batch
    
        for k in train_info.keys():
            train_info[k] /= num_updates

        y_pred, y_true = buffer.values.reshape(-1).cpu().numpy(), buffer.returns.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        train_info['losses/explained_variance'] = explained_var

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()