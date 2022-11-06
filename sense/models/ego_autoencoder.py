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
        self.deconv5 = deconv()
        convbn( 20,   40, kernel_size=7, stride=2, padding=3, bn_type=bn_type, act_type=act_type)
        self.flow5   = convbn(256,  512, kernel_size=3, stride=2, padding=1, bn_type=bn_type, act_type=act_type)
        self.deconv4 = convbn( 64,  128, kernel_size=5, stride=2, padding=2, bn_type=bn_type, act_type=act_type)
        self.flow4   = convbn(512,  512, kernel_size=3, stride=1, padding=1, bn_type=bn_type, act_type=act_type)
        self.deconv3 = convbn(128,  256, kernel_size=5, stride=2, padding=2, bn_type=bn_type, act_type=act_type)
        self.flow3   = convbn(512,  512, kernel_size=3, stride=2, padding=1, bn_type=bn_type, act_type=act_type)
        self.deconv2 = convbn(256,  256, kernel_size=3, stride=1, padding=1, bn_type=bn_type, act_type=act_type)
        self.flow2   = convbn(512,  512, kernel_size=3, stride=1, padding=1, bn_type=bn_type, act_type=act_type)
        
    def forward(self, x):
        pass