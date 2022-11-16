import numpy as np

import torch
import torch.nn as nn

from .common_v2 import *

class EGOEncoder(nn.Module):
    def __init__(self, bn_type, act_type):
        super(EGOEncoder, self).__init__()
        self.conv1    = convbn(  2,   64, kernel_size=7, stride=2, padding=3, bn_type=bn_type, act_type=act_type)
        self.conv2    = convbn( 64,  128, kernel_size=5, stride=2, padding=2, bn_type=bn_type, act_type=act_type)
        self.conv3    = convbn(128,  256, kernel_size=5, stride=2, padding=2, bn_type=bn_type, act_type=act_type)
        self.conv3_1  = convbn(256,  256, kernel_size=3, stride=1, padding=1, bn_type=bn_type, act_type=act_type)
        self.conv4    = convbn(256,  512, kernel_size=3, stride=2, padding=1, bn_type=bn_type, act_type=act_type)
        self.conv4_1  = convbn(512,  512, kernel_size=3, stride=1, padding=1, bn_type=bn_type, act_type=act_type)
        self.conv5    = convbn(512,  512, kernel_size=3, stride=2, padding=1, bn_type=bn_type, act_type=act_type)
        self.conv5_1  = convbn(512,  512, kernel_size=3, stride=1, padding=1, bn_type=bn_type, act_type=act_type)
        self.conv6    = convbn(512, 1024, kernel_size=3, stride=2, padding=1, bn_type=bn_type, act_type=act_type)
        
    def forward(self, x):
        c1 = self.conv2(self.conv1(x))
        c2 = self.conv3_1(self.conv3(c1))
        c3 = self.conv4_1(self.conv4(c2))
        c4 = self.conv5_1(self.conv5(c3))        
        c5 = self.conv6(c4)
        
        return [c1, c2, c3, c4, c5]
    

class EGODecoder(nn.Module):
    def __init__(self, bn_type, act_type):
        super(EGODecoder, self).__init__()
        self.deconv5 = deconv(1024, 512, kernel_size=4, stride=2, padding=1)
        self.flow5   = nn.Sequential( 
                            conv(1024,  2, kernel_size=3, stride=1, padding=1),
                            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.deconv4 = deconv(1024, 512, kernel_size=4, stride=2, padding=1)
        self.flow4   = nn.Sequential( 
                            conv(1026,  2, kernel_size=3, stride=1, padding=1),
                            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.deconv3 = deconv(1026, 256, kernel_size=4, stride=2, padding=1)
        self.flow3   = nn.Sequential( 
                            conv(514,  2, kernel_size=3, stride=1, padding=1),
                            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.deconv2 = deconv(514, 128, kernel_size=4, stride=2, padding=1)
        self.flow2   = nn.Sequential( 
                            conv(258,  2, kernel_size=3, stride=1, padding=1),
                            nn.Upsample(scale_factor=4, mode='bilinear'))
        
    def forward(self, x):
        c1, c2, c3, c4, c5 = x
        
        d5 = self.deconv5(c5)
        f5 = self.flow5(torch.cat((d5, c4)))
        d4 = self.deconv4(torch.cat((d5, c4)))
        f4 = self.flow4(torch.cat((d4, c3, f5)))
        d3 = self.deconv3(torch.cat((d4, c3, f5)))
        f3 = self.flow3(torch.cat((d3, c2, f4)))
        d2 = self.deconv2(torch.cat((d3, c2, f4)))
        recovered_of = self.flow2(torch.cat((d2, c1, f3)))
        
        return recovered_of