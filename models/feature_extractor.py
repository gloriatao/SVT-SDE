import torch
import torch.nn as nn
import torch.nn.functional as F
# support route shortcut
class feature_extractor(nn.Module):
    def __init__(self, num_classes=32, in_channels=3, base_chann=32, small_feat=False):
        super(feature_extractor, self).__init__()
        self.Nchannels = base_chann
        self.small_feat = small_feat
        # Initial parameters
        self.conv0 = ConvLayer_BN(in_channels, self.Nchannels, 3, 1, 1)

        self.down1 = downBlock(self.Nchannels, self.Nchannels * 2, 3, 2, 1)
        self.down2 = downBlock(self.Nchannels * 2, self.Nchannels * 4, 3, 2, 1)
        self.down3 = downBlock(self.Nchannels * 4, self.Nchannels * 8, 3, 2, 1)
        self.down4 = downBlock(self.Nchannels * 8, self.Nchannels * 16, 3, 2, 1)

        self.up3 = UpsampleBlock(self.Nchannels * 16, self.Nchannels * 8)
        self.up2 = UpsampleBlock(self.Nchannels * 8, self.Nchannels * 4)
        self.up1 = UpsampleBlock(self.Nchannels * 4, self.Nchannels * 2)
        self.up0 = UpsampleBlock(self.Nchannels * 2, self.Nchannels)

        self.conv_out = nn.Conv2d(self.Nchannels, num_classes, 1)

    def forward(self, x, return_feat=True):
        x0 = self.conv0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y3 = self.up3(x4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        if return_feat:
            y3_large = F.adaptive_avg_pool2d(y3, y2.shape[-2:])
            y1_large = F.adaptive_avg_pool2d(y1, y2.shape[-2:])
            y0_large = F.adaptive_avg_pool2d(y0, y2.shape[-2:])
            out_large = torch.cat((y3_large, y2, y1_large, y0_large), dim=1)
            return out_large
        else:
            out = self.conv_out(y0)
            return F.sigmoid(out)

class ConvLayer_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer_BN, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn(self.conv2d(x)))
        return y

class ShortcutBlock(nn.Module):
    def __init__(self, channels):
        super(ShortcutBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels//2)
        self.conv3 = nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn1(self.conv1(x)))
        y = self.leakyrelu(self.bn2(self.conv2(y)))
        y = self.leakyrelu(self.bn3(self.conv3(y)))
        y = y + x
        return y

class downBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_blocks=2):
        super(downBlock, self).__init__()
        self.down1 = ConvLayer_BN(in_channels, out_channels, kernel_size, stride, padding)
        layers = []
        for i in range(num_blocks):
            layers.append(ShortcutBlock(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(self.down1(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):

        super(UpsampleBlock, self).__init__()
        self.conv1 = ConvLayer_BN(in_channels + out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        layers = []
        for i in range(num_blocks):
            layers.append(ShortcutBlock(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, x1):
        y = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        y = torch.cat((y, x1), 1)
        y = self.conv(self.conv1(y))
        return y

