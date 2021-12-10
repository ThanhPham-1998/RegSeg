from torch.nn import functional as F
from backbone import RegsegBackbone
import torch.nn as nn
import torch

class RegsegModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        groups: int = 1,
        im_size: int = 224):
        super().__init__()

        backbone = RegsegBackbone(groups=groups,
                                  im_size=im_size)
        backbone = list(dict(backbone.named_children())['backbone'])
        self.out3 = nn.Sequential(*backbone[:2])
        self.out2 = nn.Sequential(*backbone[2:5])
        self.out1 = nn.Sequential(*backbone[5:])

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=320,
                               out_channels=128,
                               kernel_size=1,
                               bias=False)
        self.conv2a = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=1,
                                bias=False)
        self.conv2b = nn.Conv2d(in_channels=128,
                                out_channels=64,
                                kernel_size=3,
                                padding=1,
                                bias=False)
        self.conv3a = nn.Conv2d(in_channels=48,
                                out_channels=8,
                                kernel_size=1,
                                bias=False)
        self.conv3b = nn.Conv2d(in_channels=72,
                                out_channels=64,
                                kernel_size=3,
                                padding=1,
                                bias=False)
        self.conv3c = nn.Conv2d(in_channels=64,
                                out_channels=num_classes,
                                kernel_size=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.bn2a = nn.BatchNorm2d(num_features=128)
        self.bn2b = nn.BatchNorm2d(num_features=64)
        self.bn3a = nn.BatchNorm2d(num_features=8)
        self.bn3b = nn.BatchNorm2d(num_features=64)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        out3 = self.out3(x)
        out2 = self.out2(out3)
        out1 = self.out1(out2)
        out1 = self.conv1(out1)
        out1 = self.relu(self.bn1(out1))
        up1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=True)
        out2 = self.conv2a(out2)
        out2 = self.relu(self.bn2a(out2))
        up1 = F.interpolate(out1, size=(out2.size()[2], out2.size()[3]), mode='bilinear', align_corners=True)
        sum_ = up1 + out2
        out2 = self.conv2b(sum_)
        out2 = self.relu(self.bn2b(out2))
        out3 = self.conv3a(out3)
        out3 = self.relu(self.bn3a(out3))
        up2 = F.interpolate(out2, size=(out3.size()[2], out3.size()[3]), mode='bilinear', align_corners=True)
        concat = torch.cat([up2, out3], 1)
        out3 = self.conv3b(concat)
        out3 = self.relu(self.bn3b(out3))
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear', align_corners=True)
        final = self.conv3c(out3)
        return self.softmax(final)
        # return nn.Sigmoid()(final)

model = RegsegModel(im_size=500, groups=8)
x = torch.randn(1, 3, 500, 500)
print(model(x).shape)
# from torchsummary import summary
# summary(model, (3, 448, 448))

