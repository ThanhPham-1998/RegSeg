import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid())
    
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class DBlock(nn.Module):
    def __init__(
        self,
        d1: int = 1,
        d2: int = 1,
        groups: int = 1,
        in_channels: int = 3,
        out_channels: int = 1,
        stride: int = 1,
        out_size: int = 1):
        super().__init__()

        self.out_channels = out_channels
        self.conv1a = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            bias=False)
        self.conv2a = nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            bias=False)
        self.group1 = nn.Conv2d(in_channels=out_channels // 2,
                                out_channels=out_channels // 2,
                                groups=groups,
                                kernel_size=3,
                                stride=stride,
                                dilation=d1,
                                padding=d1,
                                bias=False)
        self.group2 = nn.Conv2d(in_channels=out_channels // 2,
                                out_channels=out_channels // 2,
                                groups=groups,
                                kernel_size=3,
                                stride=stride,
                                dilation=d2,
                                padding=d2,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(out_size)
        self.conv1b = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                bias=False)
        self.se_block = SEBlock(channels=out_channels,ratio=groups)
        self.bn1a = nn.BatchNorm2d(num_features=out_channels)
        self.bn_group1 = nn.BatchNorm2d(num_features=out_channels // 2)
        self.bn_group2 = nn.BatchNorm2d(num_features=out_channels // 2)
        self.bn2a = nn.BatchNorm2d(num_features=out_channels)
        self.bn1b = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, input):
        branch_a = self.relu(self.bn1a(self.conv1a(input)))
        branch_a1 = self.relu(self.bn_group1(self.group1(branch_a[:,:self.out_channels // 2,:,:])))
        branch_a2 = self.relu(self.bn_group2(self.group2(branch_a[:,self.out_channels // 2:,:,:])))
        branch_a = torch.cat([branch_a1, branch_a2], 1)
        branch_a = self.se_block(branch_a)
        branch_a = self.relu(self.bn2a(self.conv2a(branch_a)))
        branch_b = self.avgpool(input)
        branch_b = self.relu(self.bn1b(self.conv1b(branch_b)))
        final = branch_a + branch_b
        return self.relu(final)


class RegsegBackbone(nn.Module):
    def __init__(
        self, 
        groups: int = 1,
        im_size: int = 224):
        super().__init__()
        params = [
                ((1,1), 2, 32, 48, 1),
                ((1,1), 2, 48, 128, 3),
                ((1,1), 2, 128, 256, 2),
                ((1,2), 1, 256, 256, 1),
                ((1,4), 1, 256, 256, 4),
                ((1,14), 1, 256, 256, 6),
                ((1,14), 1, 256, 320, 1)
            ]
        conv = nn.Conv2d(in_channels=3,
                          out_channels=32,
                          stride=2,
                          kernel_size=3,
                          padding=1)
        backbone = []
        backbone.append(conv)
        for i, param in enumerate(params):
            if i == 0:
                size = math.ceil(im_size / 4)
            elif i == 1:
                size = math.ceil(im_size / 8)
            else:size = math.ceil(im_size / 16)

            for repeat in range(param[-1]):
                if repeat:
                    backbone.append(DBlock(d1=param[0][0],
                                           d2=param[0][1],
                                           groups=groups,
                                           in_channels=param[3],
                                           out_channels=param[3],
                                           stride=1,
                                           out_size=size))
                else:
                    backbone.append(DBlock(d1=param[0][0],
                                           d2=param[0][1],
                                           groups=groups,
                                           in_channels=param[2],
                                           out_channels=param[3],
                                           stride=param[1],
                                           out_size=size))
        self.backbone = nn.Sequential(*backbone)
    
    def forward(self, input):
        return self.backbone(input)