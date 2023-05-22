"Scalar Encoder."

import torch
import torch.nn as nn
import torch.nn.functional as F

from .libs.hyper_parameters import Arch_Hyper_Parameters as AHP
from .libs.hyper_parameters import Scalar_Feature_Size as SFS
from .libs.hyper_parameters import Action_Size as AS

debug = False

class ScalarEncoder(nn.Module):
    '''
    Inputs: scalar_features, available_actions
    Outputs:
        embedded_scalar - A 1D tensor of embedded scalar features
    '''

    def __init__(self):
        super().__init__()
        self.agent_fc = nn.Linear(SFS.num_agent_features, AHP.original_32) # with relu
        self.game_fc = nn.Linear(SFS.num_game_features, AHP.original_8) # with relu
        self.team_fc = nn.Linear(SFS.num_team_features, AHP.original_16) # with relu
        # additional features
        self.available_actions_fc = nn.Linear(AS.num_action_types, AHP.original_8) 
        
        self.fc_1 = nn.Linear(AHP.scalar_encoder_fc1_input, AHP.original_64) 
        
    
    def forward(self, scalar_tensor, available_actions):
        agent_statistics = scalar_tensor[:, :SFS.num_agent_features]
        team_statistics = scalar_tensor[:, SFS.num_agent_features:SFS.num_agent_features+SFS.num_team_features]
        game_statistics = scalar_tensor[:, SFS.num_agent_features+SFS.num_team_features:]
        
        available_actions_types = available_actions[:, :AS.num_action_types]
        
        embedded_scalar_list = []

        # agent_statistics: Embedded by assing through a linear of size 32 and a ReLU
  
        x = F.relu(self.agent_fc(agent_statistics))
        del agent_statistics
        embedded_scalar_list.append(x)


        # Teams: Both teams are embedded through a linear of size 16 and a ReLU.
        x = F.relu(self.team_fc(team_statistics))
        del team_statistics
        embedded_scalar_list.append(x)

        # Game turn: Embedded  through a linear of size 8 and a Relu
        x = F.relu(self.game_fc(game_statistics))
        del game_statistics
        embedded_scalar_list.append(x)

        # available_actions: we compute which actions may be available 
        # For example, the agent controls a worker and has enough resources to build a citytile
        # then the build action may be available 
        # The boolean vector of acton availability is passed through a linear of size 16 and a ReLu
        x = F.relu(self.available_actions_fc(available_actions_types))
        del available_actions_types
        embedded_scalar_list.append(x)
        
        embedded_scalar = torch.cat(embedded_scalar_list, dim=1)
        embedded_scalar_out = F.relu(self.fc_1(embedded_scalar))
        del x, embedded_scalar_list, embedded_scalar

        return embedded_scalar_out


def test(debug=False):
    scalar_encoder = ScalarEncoder()
    scalar_encoder = scalar_encoder.to(device="cuda")

    batch_size = 2
    # dummy scalar list
    scalar_list= []

    agent = torch.ones(batch_size, SFS.num_agent_features).to(device="cuda")
    team = torch.randn(batch_size, SFS.num_team_features).to(device="cuda")
    game = torch.randn(batch_size, SFS.num_game_features).to(device="cuda")

    available_actions = torch.randn(batch_size, SFS.num_available_actions).to(device="cuda")

    scalar_list.append(agent)
    scalar_list.append(team)
    scalar_list.append(game)

    scalar_tensor = torch.concatenate(scalar_list, dim=1)
    print("scalar_tensor", scalar_tensor) if debug else None
    print("scalar_tensor.shape:", scalar_tensor.shape) if debug else None

    embedded_scalar = scalar_encoder.forward(scalar_tensor, available_actions)

    print("embedded_scalar:", embedded_scalar) if debug else None
    print("embedded_scalar.shape:", embedded_scalar.shape) if debug else None

    if debug:
        print("This is a test!")

if __name__ == '__main__':
    test(True)