"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
from sense.lib.nn import SynchronizedBatchNorm2d
from sense.models.psmnet import PSMEncoder
from sense.models.pwc import PWCFlowDecoder, PWCDispDecoder
from .upernet import UPerNetLight

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import skimage.io
import skimage.transform
import numpy as np



class SceneNet(nn.Module):
    def __init__(self, args, num_channels=[32, 32, 64, 128, 128]
        ):
        super(SceneNet, self).__init__()
        self.encoder = PSMEncoder(args.bn_type, True)
        self.flow_decoder = PWCFlowDecoder(encoder_planes=num_channels,
                                    md=args.corr_radius,
                                    refinement_module=args.flow_refinement,
                                    bn_type=args.bn_type,
                                    pred_occ=not args.no_occ,
                                    cat_occ=args.cat_occ,
                                    upsample_output=args.upsample_flow_output)
        self.disp_decoder = PWCDispDecoder(encoder_planes=num_channels,
                                    md=args.corr_radius,
                                    do_class=args.do_class,
                                    refinement_module=args.disp_refinement,
                                    bn_type=args.bn_type,
                                    pred_occ=not args.no_occ,
                                    cat_occ=args.cat_occ
                                    )
        self.seg_decoder = UPerNetLight(
			                              num_class=args.num_seg_class, 
			                              fc_dim=num_channels[-1],
			                              fpn_inplanes=num_channels[1:], 
			                              fpn_dim=256)

        self.bn_type = args.bn_type
        self.disp_with_ppm = not args.no_ppm
        self.flow_with_ppm = not args.flow_no_ppm

        self.weight_init()

    def weight_init(self):
        if self.bn_type == 'encoding':
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, SynchronizedBatchNorm2d):
                if self.bn_type == 'syncbn':
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                else:
                    raise Exception('There should be no SynchronizedBatchNorm2d layers.')
            if isinstance(m, nn.BatchNorm2d):
                if self.bn_type == 'plain':
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                else:
                    raise Exception('There should be no nn.BatchNorm2d layers.')		

    def forward(self, input):
        for i in range(2):
          input[i] = input[i].transpose(2, 3)
        cur_l = self.encoder(input[0])
        nxt_l = self.encoder(input[1])
        if self.flow_with_ppm:
            cur_x_ = cur_l[:4] + [cur_l[-1]]
            nxt_x_ = nxt_l[:4] + [nxt_l[-1]]
        else:
            cur_x_ = cur_l[:5]
            nxt_x_ = nxt_l[:5]
        flow_multi_scale = self.flow_decoder(cur_x_, nxt_x_)

        flow = flow_multi_scale[0][0] * 20
        print_max_min(torch.max(flow).item())
        print_max_min(torch.min(flow).item())

        return flow / 400.
