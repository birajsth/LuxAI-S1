from collections import defaultdict
from ..arch.libs.hyper_parameters import Arch_Hyper_Parameters as AHP
from ..arch.libs.hyper_parameters import Action_Size as AS

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = defaultdict(lambda : None)
        for key, value in obs_space.items():
            obs_shape[key] = value.shape
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space, deterministic=True):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape

def get_available_actions_shape():
    return (AS.num_available_actions,)

def get_lstm_states_shape():
    hidden_dim=AHP.core_hidden_dim
    n_layers=AHP.lstm_layers
    return (n_layers, hidden_dim)