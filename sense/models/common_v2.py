import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sense.lib.nn import SynchronizedBatchNorm2d


def dwconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, 
                     dilation=dilation, bias=bias, groups=in_planes)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, 
                     dilation=dilation, bias=bias, groups=1)

def make_bn_layer(bn_type, plane):
    if bn_type == 'plain':
        return nn.BatchNorm2d(plane)
    elif bn_type == 'syncbn':
        return SynchronizedBatchNorm2d(plane)
    elif bn_type == 'encoding':
        raise NotImplementedError
        import encoding
        import encoding.nn
        return encoding.nn.BatchNorm2d(plane)
    else:
        raise Exception('Not supported BN type: {}.'.format(bn_type))
    
def dwconvbn(in_planes, out_planes, kernel_size=3, 
        stride=1, padding=1, dilation=1, bias=True, bn_type='syncbn', no_relu=False):
    layers = []
    layers.append(
        dwconv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=bias)
    )   
    layers.append(make_bn_layer(bn_type, out_planes))
    if not no_relu:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)

def convbn(in_planes, out_planes, kernel_size=3, 
        stride=1, padding=1, dilation=1, bias=True, bn_type='syncbn', no_relu=False
    ):
    layers = []
    layers.append(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=bias)
    )   
    layers.append(make_bn_layer(bn_type, out_planes))
    if not no_relu:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def weight_init(mdl):
    for m in mdl.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, SynchronizedBatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            
            
# pyramid pooling
# code borrowed from UPerNet
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
class PPM(nn.Module):
    def __init__(self, encoder_planes, pool_scales=(1, 2, 3, 6), bn_type='plain',
            ppm_last_conv_planes=256, ppm_inter_conv_planes=128
        ):
        super(PPM, self).__init__()
        # Parymid Pooling Module (PPM)
        self.ppm_pooling = []
        self.ppm_conv = []

        self.ppm_last_conv_planes = ppm_last_conv_planes

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(encoder_planes[-1], ppm_inter_conv_planes, kernel_size=1, bias=False),
                    make_bn_layer(bn_type, ppm_inter_conv_planes),
                    nn.ReLU(inplace=True)
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = convbn(
            encoder_planes[-1] + len(pool_scales)*128, 
            self.ppm_last_conv_planes, 
            bias=False,
            bn_type=bn_type
        )

        weight_init(self)

    def forward(self, conv5):
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False
            )))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        return f
