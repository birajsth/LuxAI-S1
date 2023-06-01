" Action Type Head."


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .libs.hyper_parameters import Arch_Hyper_Parameters as AHP
from .libs.hyper_parameters import Action_Size as AS


debug = False


class ActionHead(nn.Module):
    '''
    Inputs: core_output, available_actions
    Outputs:
        action_log_prob - The log probabilities of taking each action
        action entropy - The entropy in action probabilities 
        action_type - The action_type sampled from the action_type_logits
    '''

    def __init__(self, core_dim=AHP.core_hidden_dim,
                 hidden_size=128,  
                 use_action_mask=AHP.use_action_type_mask):
        super().__init__()

        self._use_action_mask = use_action_mask
    
        self.embed_fc = nn.Linear(core_dim, hidden_size) 
        self.hidden_fc_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_fc_2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_fc_3 = nn.Linear(hidden_size, hidden_size)

        self.out_action_type = nn.Linear(hidden_size, AS.num_action_types)
        self.out_move_direction = nn.Linear(hidden_size, AS.num_move_directions)
        self.out_transfer_direction = nn.Linear(hidden_size, AS.num_transfer_directions)

    

    def forward(self, core_output, available_actions, action=None, deterministic=False):
        device = next(self.parameters()).device
        available_action_type = available_actions[:, :AS.num_action_types]
        available_move_direction = available_actions[:, AS.num_action_types:AS.num_action_types + AS.num_move_directions]
        available_transfer_direction = available_actions[:, AS.num_action_types + AS.num_move_directions: AS.num_action_types + AS.num_move_directions + AS.num_transfer_directions]
        del available_actions
        x = F.relu(self.embed_fc(core_output))
        del core_output

        x_action_type = F.relu(self.hidden_fc_1(x))
        x_move_direction = F.relu(self.hidden_fc_2(x))
        x_transfer_direction = F.relu(self.hidden_fc_3(x))

        del x

        action_type_logits = self.out_action_type(x_action_type)
        del x_action_type
        
        # inspired by the DI-star project, in action_type_head
        if self._use_action_mask:
            action_type_mask = available_action_type.bool()
            action_type_logits =  torch.where(action_type_mask, action_type_logits, torch.tensor(-1e+8).to(device))
            del action_type_mask

        
        action_type_probs = Categorical(logits=action_type_logits)
        if action is None:
            action_type = torch.argmax(action_type_logits, dim=1) if deterministic else action_type_probs.sample()
        else: 
            action_type = action[:,0]

        # action_direction_head
        move_direction_logits = self.out_move_direction(x_move_direction)
        transfer_direction_logits = self.out_transfer_direction(x_transfer_direction)
        del x_move_direction, x_transfer_direction

        # inspired by the DI-star project, in action_type_head
        if self._use_action_mask:
            move_direction_mask = available_move_direction.bool()
            transfer_direction_mask = available_transfer_direction.bool()
            # set default action True if the corresponding action is not to be taken
            #move_direction_mask[:, 4] = ~(action_type==0)
            #transfer_direction_mask[:, 4] = ~(action_type==1)
            move_direction_logits = torch.where(move_direction_mask, move_direction_logits, torch.tensor(-1e+8).to(device))
            transfer_direction_logits = torch.where(transfer_direction_mask, transfer_direction_logits, torch.tensor(-1e+8).to(device))
            del move_direction_mask, transfer_direction_mask

        move_direction_probs = Categorical(logits=move_direction_logits)
        transfer_direction_probs = Categorical(logits=transfer_direction_logits)
        if action is None:
            move_direction = torch.argmax(move_direction_logits, dim=1) if deterministic else move_direction_probs.sample()
            #move_direction = torch.where(action_type==0, move_direction, torch.tensor(4).to(device))
            transfer_direction = torch.argmax(transfer_direction_logits, dim=1) if deterministic else transfer_direction_probs.sample()
            #transfer_direction = torch.where(action_type==1, transfer_direction, torch.tensor(4).to(device))
        else: 
            move_direction = action[:,1]
            transfer_direction = action[:,2]

        move_direction_logprobs = torch.where(action_type==0, move_direction_probs.log_prob(move_direction), torch.tensor(0).to(device))
        transfer_direction_logprobs = torch.where(action_type==1, transfer_direction_probs.log_prob(transfer_direction), torch.tensor(0).to(device))

        action_log_probs = action_type_probs.log_prob(action_type) + move_direction_logprobs + transfer_direction_logprobs
        action_entropy = action_type_probs.entropy() + move_direction_probs.entropy() + transfer_direction_probs.entropy() 
        action = torch.cat((action_type.unsqueeze(-1), move_direction.unsqueeze(-1), transfer_direction.unsqueeze(-1)),dim=1)

        del action_type_logits, move_direction_logits, transfer_direction_logits
        del action_type_probs, move_direction_probs, transfer_direction_probs
        del move_direction_logprobs, transfer_direction_logprobs
        del action_type, move_direction, transfer_direction

        return action_log_probs, action_entropy, action

def test(debug=False):
    batch_size = 2
    core_output = torch.randn(batch_size, AHP.core_hidden_dim)

    available_actions = torch.zeros(batch_size, AS.num_available_actions)
    available_actions[:,[0,3]] = 1
    action = torch.zeros(batch_size, 4)
    action[:,0] = 1
    action_type_head = ActionHead()

    print("core_output:", core_output) if debug else None
    print("core_output.shape:", core_output.shape) if debug else None
    
    action_logprob, _, action_type = action_type_head.forward(core_output, available_actions, action, deterministic=True)

    print("action_logprobs:", action_logprob) if debug else None
    print("action_logprobs.shape:", action_logprob.shape) if debug else None
    print("action:", action_type) if debug else None
    print("action.shape:", action_type.shape) if debug else None
    
    print("This is a test!") if debug else None

if __name__=="__main__":
    test(True)