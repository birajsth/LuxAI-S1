import torch.nn as nn

from .libs.param_initalize import init_weights

from .scalar_encoder import ScalarEncoder
from .spatial_encoder import SpatialEncoder
from .core import Core
from .action_head import ActionHead
from .baseline import Baseline

class Actor(nn.Module):
    def __init__(self, use_lstm=True, use_squeeze_excitation=True):
        super(Actor, self).__init__()

        self.scalar_encoder = ScalarEncoder()
        self.spatial_encoder = SpatialEncoder(squeeze_excitation=use_squeeze_excitation)
        self.core = Core(use_lstm=use_lstm)
        self.out = ActionHead()
        # init all parameters
        self.apply(init_weights)

    def forward(self, x_scalar, x_spatial, x_available_actions, hidden_state=None, action=None, deterministic=False):
        embedded_scalar = self.scalar_encoder(x_scalar, x_available_actions)
        embedded_spatial = self.spatial_encoder(x_spatial)

        core_out, hidden_state = self.core(embedded_scalar, embedded_spatial, hidden_state)

    
  
        action_log_prob, entropy, action = self.out(core_out, x_available_actions, action, deterministic)
        return action, action_log_prob, entropy, hidden_state
    
    
    
class Critic(nn.Module):
    def __init__(self, use_lstm=False, use_squeeze_excitation=True, use_popart=True):
        super(Critic, self).__init__()

        self.scalar_encoder = ScalarEncoder()
        self.spatial_encoder = SpatialEncoder(squeeze_excitation=use_squeeze_excitation)
        self.core = Core(use_lstm=use_lstm)
        self.out = Baseline(use_popart=use_popart)
        # init all parameters
        self.apply(init_weights)


    def forward(self, x_scalar, x_spatial, x_available_actions, hidden_state=None):
        embedded_scalar = self.scalar_encoder(x_scalar, x_available_actions)
        embedded_spatial = self.spatial_encoder(x_spatial)

        core_out, hidden_state = self.core(embedded_scalar, embedded_spatial, hidden_state)
        return self.out(core_out), hidden_state
    

