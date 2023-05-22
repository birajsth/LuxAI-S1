import torch
import torch.nn as nn
import torch.nn.functional as F



class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Average feature planes
        y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    # without batchnorm
    def __init__(self, inplanes=32, planes=32, stride=1, squeeze_excitation=True) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.relu = nn.ReLU()

        if squeeze_excitation:
            self.squeeze_excitation = SELayer(planes)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x):
        z = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.squeeze_excitation(x)

        return x + z
    


class ResBlock1D(nn.Module):

    def __init__(self, inplanes, planes, seq_len, 
                 stride=1, downsample=None, norm_type=None):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # Layer norm is applied in the not N channels, 
        # e.g., E of (N, S, E) in NLP,
        # and [C, H, W] of (N, C, H, W) in CV.
        # Note, because Layer norm doesn't average over dim of batch (N), so
        # it is the same in training and evaluating.
        # For Batch Norm, this is not the case.
        #self.ln1 = nn.LayerNorm([planes, seq_len])
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.normtype = norm_type

    def forward(self, x):
        if self.normtype == 'prev':
            residual = x
            x = F.relu(self.ln1(x))
            x = self.conv1(x)
            x = F.relu(self.ln2(x))
            x = self.conv2(x)
            x = x + residual
            del residual
            return x
        elif self.normtype == 'post':
            residual = x
            x = F.relu(self.conv1(x))
            x = self.ln1(x)
            x = F.relu(self.conv2(x))
            x = self.ln2(x)
            x = x + residual
            del residual
            return x
        else:
            residual = x
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x + residual
            del residual
            return x
        
if __name__ == '__main__':
    pass