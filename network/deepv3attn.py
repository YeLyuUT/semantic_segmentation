"""
Code Adapted from:
https://github.com/sthalles/deeplab_v3

Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn

from network.mynn import initialize_weights, Norm2d, Upsample
from network.utils import get_aspp, get_trunk, make_seg_head
from network.apnb import APNB
from network.afnb import AFNB


class DeepV3PlusATTN(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 use_dpc=False, init_all=False):
        super(DeepV3PlusATTN, self).__init__()
        self.criterion = criterion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        #self.aspp, aspp_out_ch = get_aspp(high_level_ch,
        #                                  bottleneck_ch=256,
        #                                  output_stride=8,
        #                                  dpc=use_dpc)
        #self.attn = APNB(in_channels=high_level_ch, out_channels=high_level_ch, key_channels=256, value_channels=256, dropout=0.5, sizes=([1]), norm_type='batchnorm', psp_size=(1, 3, 6, 8))
        self.attn = AFNB(low_in_channels = 1024, high_in_channels=4096, out_channels=1024, key_channels=256, value_channels=1024, dropout=0.5, sizes=([1]), norm_type='batchnorm',psp_size=(1,3,6,8))
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        if init_all:
            initialize_weights(self.attn)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        x_size = x.size()
        s2_features, s4_features, final_features = self.backbone(x)
        aspp = self.attn(s4_features, final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(out, gts)

        return {'pred': out}


def DeepV3PlusATTNSRNX50(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='seresnext-50', criterion=criterion)


def DeepV3PlusATTNR50(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='resnet-50', criterion=criterion)


def DeepV3PlusATTNSRNX101(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='seresnext-101', criterion=criterion)


def DeepV3PlusATTNW38(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='wrn38', criterion=criterion)


def DeepV3PlusATTNW38I(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='wrn38', criterion=criterion,
                      init_all=True)


def DeepV3PlusATTNX71(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='xception71', criterion=criterion)


def DeepV3PlusATTNEffB4(num_classes, criterion):
    return DeepV3PlusATTN(num_classes, trunk='efficientnet_b4',
                      criterion=criterion)


class DeepV3ATTN(nn.Module):
    """
    DeepLabV3 with various trunks supported
    """
    def __init__(self, num_classes, trunk='resnet-50', criterion=None,
                 use_dpc=False, init_all=False, output_stride=8):
        super(DeepV3ATTN, self).__init__()
        self.criterion = criterion

        self.backbone, _s2_ch, _s4_ch, high_level_ch = \
            get_trunk(trunk, output_stride=output_stride)
        #self.aspp, aspp_out_ch = get_aspp(high_level_ch,
        #                                  bottleneck_ch=256,
        #                                  output_stride=output_stride,
        #                                  dpc=use_dpc)
        #self.attn = APNB(in_channels=high_level_ch, out_channels=high_level_ch, key_channels=256, value_channels=256, dropout=0.5, sizes=([1]), norm_type='batchnorm', psp_size=(1,3,6,8))
        self.attn = AFNB(low_in_channels = 1024, high_in_channels=4096, out_channels=1024, key_channels=1024, value_channels=4096, dropout=0.5, sizes=([1]), norm_type='batchnorm',psp_size=(1,3,6,8))
        self.final = make_seg_head(in_ch=high_level_ch, out_ch=num_classes)

        initialize_weights(self.attn)
        initialize_weights(self.final)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        x_size = x.size()
        s2_features, s4_features, final_features = self.backbone(x)
        attn = self.attn(s4_features, final_features)
        final = self.final(attn)
        out = Upsample(final, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(out, gts)

        return {'pred': out}


def DeepV3ATTNR50(num_classes, criterion):
    return DeepV3ATTN(num_classes, trunk='resnet-50', criterion=criterion)

