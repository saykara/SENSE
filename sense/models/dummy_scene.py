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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import skimage.io
import skimage.transform
import numpy as np

from sense.rigidity_refine.rigidity_refine import warp_disp_refine_rigid


class DummySceneNet(nn.Module):
    def __init__(self,args, flow_dec=None, disp_dec=None, 
            seg_dec=None, bn_type='plain',
            disp_with_ppm=False, flow_with_ppm=False
        ):
        super(DummySceneNet, self).__init__()
        self.encoder = PSMEncoder(args.bn_type, True)
        self.flow_decoder = PWCFlowDecoder(encoder_planes=[32, 32, 64, 128, 128],
                                    md=args.corr_radius,
                                    refinement_module=args.flow_refinement,
                                    bn_type=args.bn_type,
                                    pred_occ=not args.no_occ,
                                    cat_occ=args.cat_occ,
                                    upsample_output=args.upsample_flow_output)
        self.disp_decoder = PWCDispDecoder(encoder_planes=[32, 32, 64, 128, 128],
                                    md=args.corr_radius,
                                    do_class=args.do_class,
                                    refinement_module=args.disp_refinement,
                                    bn_type=args.bn_type,
                                    pred_occ=not args.no_occ,
                                    cat_occ=args.cat_occ
                                    )
        self.seg_decoder = None
        assert flow_dec is not None \
               or disp_dec is not None \
               or seg_dec is not None, \
               'at least one of the decoders should not be None'

        self.bn_type = bn_type
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
        cur_l = self.encoder(input[0])
        cur_r = self.encoder(input[1])
        nxt_l = self.encoder(input[2])
        nxt_r = self.encoder(input[3])

        if self.flow_with_ppm:
            cur_x_ = cur_l[:4] + [cur_l[-1]]
            nxt_x_ = nxt_l[:4] + [nxt_l[-1]]
        else:
            cur_x_ = cur_l[:5]
            nxt_x_ = nxt_l[:5]
        flow_multi_scale = self.flow_decoder(cur_x_, nxt_x_)

        
        if self.disp_with_ppm:
            cur_x_ = cur_l[:4] + [cur_l[-1]]
            left_x_ = cur_r[:4] + [cur_r[-1]]
        else:
            cur_x_ = cur_l[:5]
            left_x_ = cur_r[:5]
        disp0 = self.disp_decoder(cur_x_, left_x_)
        
        
        if self.disp_with_ppm:
            left_x_ = nxt_l[:4] + [nxt_l[-1]]
            right_x_ = nxt_r[:4] + [nxt_r[-1]]
        else:
            left_x_ = nxt_l[:5]
            right_x_ = nxt_r[:5]
        disp1 = self.disp_decoder(left_x_, right_x_)
   
        K0, K1 = input[4]
        flow_rigid, _, _ = warp_disp_refine_rigid(disp0, disp1, flow_multi_scale, None, K0, K1)

        return flow_rigid / 400.
