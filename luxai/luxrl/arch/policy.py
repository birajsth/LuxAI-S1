import numpy as np
import torch
import torch.nn as nn

from .libs.utils import update_linear_schedule
from .actor_critic import Actor, Critic


class PolicyN(nn.Module):
    def __init__(self, args):
        super(PolicyN, self).__init__()
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.use_popart=args.use_popart
        self.use_lstm=args.use_lstm
        self.use_squeeze_excitation=args.use_squeeze_excitation
        
        self.actor = Actor(self.use_lstm, self.use_squeeze_excitation)
        self.critic = Critic(use_squeeze_excitation= self.use_squeeze_excitation, use_popart=self.use_popart)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    def lr_decay(self, update, num_updates):
        """
        Decay the actor and critic learning rates.
        :param update: (int) current training update.
        :param num_updates: (int) total number of training updates.
        """
        update_linear_schedule(self.actor_optimizer, update, num_updates, self.lr)
        update_linear_schedule(self.critic_optimizer, update, num_updates, self.critic_lr)

    def get_value(self, x_scalar, x_spatial, x_available_actions, hidden_state=None):
        value, _ = self.critic(x_scalar, x_spatial, x_available_actions, hidden_state)
        return value

    def get_action_and_value(self, x_scalar, x_spatial, x_available_actions, hidden_state=None, action=None, deterministic=False):
        action, action_log_prob, entropy, hidden_state = self.actor(x_scalar, x_spatial, x_available_actions, hidden_state, action, deterministic)
        value, _ = self.critic(x_scalar, x_spatial, x_available_actions)
        return action, action_log_prob, entropy, value, hidden_state
    
    def predict(self, x_scalar, x_spatial, x_available_actions, hidden_state=None, deterministic=True):
        if type(x_scalar) == np.ndarray:
            device = next(self.parameters()).device
            x_scalar = torch.from_numpy(x_scalar).unsqueeze(0).to(device)
            x_spatial = torch.from_numpy(x_spatial).unsqueeze(0).to(device)
            x_available_actions = torch.from_numpy(x_available_actions).unsqueeze(0).to(device)
        action, _, _, hidden_state = self.actor(x_scalar, x_spatial, x_available_actions, hidden_state, deterministic=deterministic)
        del x_scalar, x_available_actions, x_spatial, _
        return action, hidden_state
    