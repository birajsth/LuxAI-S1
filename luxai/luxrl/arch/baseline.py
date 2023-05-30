" Value Head"
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .libs.resblock import ResBlock1D
from .libs.popart import PopArt
from .libs import utils as L
from .libs.hyper_parameters import Arch_Hyper_Parameters as AHP

debug = False


class Baseline(nn.Module):
    '''    
    Inputs: core_output
    Outputs:
        winloss_baseline - A baseline value 
    '''

    def __init__(self, 
                 baseline_input=AHP.core_hidden_dim, 
                 hidden_size=AHP.original_128,
                 use_popart=False,
                 ):
        super().__init__()
        self._use_popart = use_popart
        self.hidden_size = hidden_size
        
        #MLP
        self.embed_fc = nn.Linear(baseline_input, hidden_size)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size) 
        
        if self._use_popart:
            self.v_out = PopArt(self.hidden_size, 1)
        else:
            self.v_out = nn.Linear(self.hidden_size, 1)

        
    def forward(self, core_output):
        # MLP
        x = F.relu(self.embed_fc(core_output))
        x = F.relu(self.hidden_fc(core_output))
        del core_output
        
        if self._use_popart:
            out = self.v_out(x)
        else:
            out = self.v_out(x)
            print("x.shape:", x.shape) if debug else None
            # This baseline value is transformed by ((2.0 / PI) * atan((PI / 2.0) * baseline)) and is used as the baseline value
            #out = (2.0 / np.pi) * torch.atan((np.pi / 2.0) * out)
            print("out:", out) if debug else None
        del x

        return out


def test(debug=False):
    baseline = Baseline().to(device="cuda")
    batch_size = 2

    core_output = torch.ones(batch_size, AHP.core_hidden_dim).to(device="cuda")

    out = baseline.forward(core_output)

    print("out:", out) if debug else None
    print("out.shape:", out.shape) if debug else None

    if debug:
        print("This is a test!")

if __name__ == '__main__':
    test(True)