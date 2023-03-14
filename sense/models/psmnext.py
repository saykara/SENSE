"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import DropPath

from sense.lib.nn import SynchronizedBatchNorm2d

from .common_v2 import *

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, stride, downsample, pad, dilation, bn_type, is_drop):
        super(BasicBlock, self).__init__()

        self.dwconv = dwconv(inplanes, planes, kernel_size, stride, pad, dilation)
        self.bn = make_bn_layer(bn_type, planes)
        self.linear1 = conv(planes, 4 * planes, 1, 1, 0, 1)
        self.act = nn.GELU()
        self.linear2 = conv(4 * planes, planes, 1, 1, 0, 1)

        self.downsample = downsample
        self.stride = stride
        self.is_drop = is_drop

    def forward(self, x):
        out = self.dwconv(x)
        out = self.bn(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        if self.is_drop:
            result = x + self.downsample(out)
        else:
            result = self.downsample(x) + out
        return result

class PSMNextEncoder(nn.Module):
    def __init__(self, bn_type, kernel_size=3, with_ppm=False):
        super(PSMNextEncoder, self).__init__()
        self.inplanes = 32
        pad = (int)((kernel_size - 1) / 2)
        self.firstconv = nn.Sequential(
                        convbn(3,  32, 3, 2, 1, 1, bn_type=bn_type),
                        convbn(32, 32, 3, 1, 1, 1, bn_type=bn_type),
                        convbn(32, 32, 3, 1, 1, 1, bn_type=bn_type)
                        )
        self.layer1 = self._make_layer(BasicBlock, 32, 3, kernel_size, 2,pad,1, bn_type)
        self.layer2 = self._make_layer(BasicBlock, 64, 3, kernel_size, 2,pad,1, bn_type) 
        self.layer3 = self._make_layer(BasicBlock, 128, 9, kernel_size, 2,pad,1, bn_type)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, kernel_size, 2,pad,1, bn_type)

        if with_ppm:
            self.ppm = PPM(
                [32, 32, 64, 128, 128],
                ppm_last_conv_planes=128,
                ppm_inter_conv_planes=128,
                bn_type=bn_type
                )
        else:
            self.ppm = None

    def _make_layer(self, block, planes, blocks, kernel_size, stride, pad, dilation, bn_type):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                make_bn_layer(bn_type, self.inplanes),
                conv(in_planes=self.inplanes, out_planes=planes * block.expansion, 
                       kernel_size=2, stride=stride, padding=0, bias=False))

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample, pad, dilation, bn_type, False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, 1, DropPath(0.), pad, dilation, bn_type, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        output1      = self.firstconv(x)
        output2      = self.layer1(output1)
        output3      = self.layer2(output2)
        output4      = self.layer3(output3)
        output5      = self.layer4(output4)

        if self.ppm is not None:
            output5_2 = self.ppm(output5)
        else:
            output5_2 = None

        return [output1, output2, output3, output4, output5, output5_2]

if __name__ == '__main__':
    enc = PSMNextEncoder('plain')
    for n, m in enc.named_modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(n)

    # x = torch.zeros((2, 3, 256, 256))
    # y = enc(x)
    # for y_ in y:
    #     print(y_.size())
