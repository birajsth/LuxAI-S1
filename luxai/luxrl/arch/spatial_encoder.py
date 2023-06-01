"""Spatial Encoder"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .libs.hyper_parameters import Spatial_Feature_Size as SPFS
from .libs.hyper_parameters import Arch_Hyper_Parameters as AHP
from .libs.resblock import ResBlock


debug = False

class SpatialEncoder(nn.Module):
    '''
    Inputs: map
    Outputs: 
       embedded_spatial - A 1D tensor of the embedded map
    '''
    
    def __init__(self, n_resblocks=AHP.n_resblocks, 
                 hidden_size=32, output_size=64,
                 squeeze_excitation=True
                 ) -> None:
        super().__init__()
        self.use_improved_one = True

        self.project_inplanes = SPFS.num_spatial_features
        self.project = nn.Conv2d(self.project_inplanes, hidden_size, kernel_size=3, stride=1,
                                padding=1, bias=True)
        
        self.resblock_stack = nn.ModuleList([
            ResBlock(inplanes=hidden_size, planes=hidden_size, stride=1, squeeze_excitation=squeeze_excitation)
            for _ in range(n_resblocks)])
        
        self.fc = nn.Linear(11 * 11 * hidden_size, output_size)
    
    
    
    def forward(self, x):
        # the planes are projected to 32 channels 
        # by a 2D convolution with kernel size 3, padding 1, passed through a ReLU
        x = F.relu(self.project(x))
        
        # 4 ResBlocks with 32 channels and kernel size 3 and applied to the projected map, 
        for resblock in self.resblock_stack:
            x = resblock(x)

        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        # The ResBlock output is embedded into a 1D tensor of size 64 by a linear layer 
        # and a ReLU, which becomes `embedded_spatial`.
        x = self.fc(x)
        embedded_spatial = F.relu(x)
        del x
        return embedded_spatial
    

        

def test(debug=False):
    spatial_encoder = SpatialEncoder()
    batch_size = 2

    #dummy map list
    map_list = []

    map_data_1 = torch.zeros(batch_size, SPFS.num_spatial_features, SPFS.width, SPFS.height)
    map_list.append(map_data_1)
    map_data_2 = torch.zeros(batch_size, SPFS.num_spatial_features, SPFS.width, SPFS.height)
    map_list.append(map_data_2)

    map_data = torch.cat(map_list, dim=0)

    embedded_spatial = spatial_encoder.forward(map_data)

    print('embedded_spatial:', embedded_spatial) if debug else None
    print('embedded_spatial.shape:', embedded_spatial.shape) if debug else None

