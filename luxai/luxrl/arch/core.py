
" Core."

import torch
import torch.nn as nn
import torch.nn.functional as F

from .libs.hyper_parameters import Arch_Hyper_Parameters as AHP


debug = False


class Core(nn.Module):
    '''
    Inputs: embedded_spatial, embedded_scalar
    Outputs:
    
    '''
    def __init__(self, embedding_dim=AHP.original_128, hidden_dim=AHP.core_hidden_dim, 
                 n_layers=AHP.lstm_layers, use_lstm=True, drop_prob=0.0) -> None:
        super(Core, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self._use_lstm=use_lstm

        if self._use_lstm:
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, 
                            dropout=drop_prob, batch_first=True)
        else: # MLP
            self.fc = nn.Linear(AHP.original_128, AHP.original_128)
        
        

    def forward(self, embedded_scalar, embedded_spatial, hidden_state=None):
        combined = torch.cat([embedded_scalar, embedded_spatial], dim=1)
        del embedded_scalar, embedded_spatial
        
        if self._use_lstm:
            # we transform the shape from [batch_size, embedding_size] 
            # to [batch_size, seq_size=1, embedding_size] 
            combined = combined.unsqueeze(1)
            
            if hidden_state is None:
                hidden_state = self.init_hidden_state(batch_size=combined.shape[0])
            # Changing the shape from [batch_size, n_layers, hidden_dim] to [n_layers, batch_size, hidden_dim]
            hidden_state = (hidden_state[0].permute(1,0,2).contiguous(), hidden_state[1].permute(1,0,2).contiguous())


            lstm_output, hidden_state = self.lstm(combined, hidden_state)
            # Back to [batch_size, embedding_size] 
            print("lstm output: shape ", lstm_output.shape) if debug else None
            lstm_output = lstm_output.squeeze(1)
            # Changing the shape from [n_layers, batch_size, hidden_dim] to [batch_size, n_layers, hidden_dim]
            hidden_state = (hidden_state[0].permute(1,0,2), hidden_state[1].permute(1,0,2))

            del combined
            print("hidden_state shape: ", hidden_state[0].shape) if debug else None
            return lstm_output, hidden_state
        
        #MLP
        combined = F.relu(self.fc(combined))
        return combined, None
        
        
    
    def init_hidden_state(self, batch_size=1):
        device = next(self.parameters()).device
        hidden = (torch.zeros(batch_size, self.n_layers, self.hidden_dim).to(device), 
                  torch.zeros(batch_size, self.n_layers, self.hidden_dim).to(device))

        return hidden


