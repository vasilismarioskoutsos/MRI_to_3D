import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvolution(nn.Module):
    # has the 2 3x3 convs and the relu layer
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.padding = 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.padding) # double the channels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=self.padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.batch_norm(self.conv1(x))
        x = self.relu(x)
        x = self.batch_norm(self.conv2(x))
        x = self.relu(x)
        return x
    
class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = DoubleConvolution(in_channels, out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(self.pooling(x))
        return x
    
class ExpandingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = DoubleConvolution(in_channels, out_channels)
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        # x1 = input feature map, x2 = skip connection feature map
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        # pad in case of size convonsistencies 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # merge the 2 feature into a new dimension
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)
    
class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.out(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.conv = DoubleConvolution(in_channels, 64)
        self.down1 = ContractingPath(64, 128)
        self.down2 = ContractingPath(128, 256)
        self.down3 = ContractingPath(256, 512)
        self.down4 = ContractingPath(512, 1024)
        self.up1 = ExpandingPath(1024, 512)
        self.up2 = ExpandingPath(512, 256)
        self.up3 = ExpandingPath(256, 128)
        self.up4 = ExpandingPath(128, 64)
        self.out = OutConvolution(64, out_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x