import torch
import torch.nn as nn
import numpy as np

# Weights inititalization
def init_weights(module, std=np.sqrt(2), bias_const=0.0):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data, gain=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            torch.nn.init.constant_(module.bias.data, bias_const)
    elif isinstance(module, nn.Conv1d) or \
            isinstance(module, nn.Conv2d):
        torch.nn.init.orthogonal_(module.weight.data, std)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias.data, bias_const)
    else:
        if hasattr(module, 'weight'):
            if module.weight is not None: 
                if module.weight.data.dim() > 1:
                    nn.init.xavier_uniform_(module.weight.data)
        if hasattr(module, 'bias'):
            if module.bias is not None: 
                if hasattr(module.bias, 'data'):
                    if module.bias.data.dim() > 1:
                        nn.init.constant_(module.bias.data, 0)