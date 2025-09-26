import torch
import torch.functional as F
import torch.nn as nn


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class CNNEncoder(nn.Module):
    arch_settings = {
        9: (ResBlock, (3, 3, 2))
    }

    def __init__(self, depth, in_channels=1, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, dims=[192, 384, 384]):
        super(CNNEncoder, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.layer1 = self._make_layer(block, dims[0], layers[0], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, dims[1], layers[1], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, dims[2], layers[2], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []

        # 反卷积
        # self.upconv1 = nn.ConvTranspose3d(384, 384, kernel_size=2, stride=(2, 2, 2))
        # self.conv4 = nn.Conv3d(384, 384, kernel_size=3, padding=1)
        # self.upconv2 = nn.ConvTranspose3d(384, 192, kernel_size=2, stride=(2, 2, 2))
        # self.conv5 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        # self.upconv3 = nn.ConvTranspose3d(192, 192, kernel_size=2, stride=(2, 2, 2))
        # self.upconv4 = nn.Upsample(scale_factor=(1, 2, 2))
        # self.conv6 = nn.Conv3d(192, 1, kernel_size=1)


        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU', weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False, weight_std=weight_std), Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        print("input shape:", x.shape)  # Batch Channel Depth Height Weight
        x = self.conv1(x)   # [2, 1, 48, 192, 192]--> [2, 64, 48, 96, 96]
        x = self.norm1(x)
        x = self.nonlin(x)

        x1 = self.layer1(x)  # [2, 64, 48, 96, 96]--> [2, 192, 24, 48, 48]
        x2 = self.layer2(x1)  # [2, 192, 24, 48, 48]--> [2, 384, 12, 24, 24]
        x3 = self.layer3(x2)  # [2, 384, 12, 24, 24]--> [2, 384, 6, 12, 12]

        # x4 = self.upconv1(x3)  # [2, 384, 6, 12, 12]--> [2, 384, 12, 24, 24]
        # print('x4:', x4.shape)
        # x4 = x4 + x2
        # print('x4:', x4.shape)
        # x4 = torch.relu(self.conv4(x4))
        #
        # x5 = self.upconv2(x4)  # [2, 384, 12, 24, 24] --> [2, 192, 24, 48, 48]
        # print('x5:', x5.shape)
        # x5 = x5 + x1
        # print('x5:', x5.shape)
        # x5 = torch.relu(self.conv5(x5))
        #
        #
        # x6 = self.upconv3(x5)  # [2, 192, 24, 48, 48] --> [2, 192, 48, 96, 96]
        # print('x6:', x6.shape)
        # # x7 = self.conv6(x6)
        # x7 = self.upconv4(x6)
        # print('x7:', x7.shape)
        # x7 = self.conv6(x7)
        # # return out  # out包含了4个分辨率的feature map
        return x3