import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from model_zoo.padding import ReplicationPad3d

from .common import BaseNetwork


class InpaintGenerator(BaseNetwork):
    def __init__(self, rates='1+2+4+8', block_num=8):  
        nr_channels = 1
        rates=[1, 2, 4, 8]
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            ReplicationPad3d(3),
            nn.Conv3d(2, 64, 7),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv3d(64, nr_channels, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        
        x = self.encoder(x)
        
        x = self.middle(x)
        
        x = self.decoder(x)
        
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv3d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    ReplicationPad3d(rate),
                    nn.Conv3d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            ReplicationPad3d(1),
            nn.Conv3d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            ReplicationPad3d(1),
            nn.Conv3d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        
        out = self.fuse(out)
        
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat





class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 1
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat

