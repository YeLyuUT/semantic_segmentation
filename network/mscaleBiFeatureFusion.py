"""
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

from config import cfg
from network.mynn import initialize_weights, Norm2d, Upsample, Upsample2
from network.mynn import ResizeX, scale_as
from network.utils import get_aspp, get_trunk, ConvBnRelu
from network.utils import make_seg_head, make_attn_head
from utils.misc import fmt_scale

class MscaleBase(nn.Module):
    """
    Multi-scale attention segmentation model base class
    """
    def __init__(self):
        super(MscaleBase, self).__init__()
        self.criterion = None
        self.fuse_aspp = False

    def _fwd(self, x, aspp_in=None):
        pass

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred1 = None
        output_dict = {}
        for s in scales:
            x = ResizeX(x_1x, s)
            bs = x.shape[0]
            scale_float = torch.Tensor(bs).fill_(s)
            _p, aspp, cat_s4_attn = self._fwd_feature(x)
            attn, aspp_attn = self._fwd_attn(x, cat_s4_attn)
            output_dict[fmt_scale('pred', s)] = _p
            output_dict[fmt_scale('cat_s4_attn', s)] = cat_s4_attn
            c = _p.shape[1]
            p = _p.narrow(1,0,int(c/2))
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn

            if pred1 is None:
                pred1 = p
            elif s >= 1.0:
                # downscale previous
                pred1 = scale_as(pred1, p)
                pred1 = attn * p + (1 - attn) * pred1
            else:
                # upscale current
                p = attn * p
                p = scale_as(p, pred1)
                attn = scale_as(attn, pred1)
                pred1 = p + (1 - attn) * pred1

        pred2 = None
        for s in scales[::-1]:
            _p = output_dict[fmt_scale('pred', s)]
            cat_s4_attn = output_dict[fmt_scale('cat_s4_attn', s)]
            attn, aspp_attn = self._fwd_attn_rev(_p, cat_s4_attn)
            c = _p.shape[1]
            p = _p.narrow(1,int(c/2),int(c/2))
            if pred2 is None:
                pred2 = p
            elif s > 1.0:
                # downscale previous
                p = attn * p
                p = scale_as(p, pred2)
                attn = scale_as(attn, pred2)
                pred2 = p + (1 - attn) * pred2
            else:
                # upscale current
                pred2 = scale_as(pred2, p)
                pred2 = attn * p + (1 - attn) * pred2

        #pred = (pred1+pred2)/2.0
        pred = torch.cat([pred1, pred2], dim=1)
        pred = self.classifier(pred)

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(pred, gts)
            return loss
        else:
            output_dict['pred'] = pred
            return output_dict

    def two_scale_forward(self, inputs):
        assert 'images' in inputs
        x_1x = inputs['images']
        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)

        pred_05x, aspp_lo, cat_s4_attn_lo = self._fwd_feature(x_lo)
        p_1x, aspp_1x, cat_s4_attn_1x = self._fwd_feature(x_1x)
        attn_lo, aspp_attn_lo = self._fwd_attn(x_lo, cat_s4_attn_lo)
        attn_1x, aspp_attn_1x = self._fwd_attn_rev(x_1x, cat_s4_attn_1x)
        #low 2 high
        c = attn_lo.shape[1]
        p_lo = attn_lo * pred_05x.narrow(1,0,int(c/2))
        p_lo = scale_as(p_lo, p_1x)
        logit_attn = scale_as(attn_lo, p_1x)
        joint_pred1 = p_lo + (1 - logit_attn) * p_1x
        #high 2 low
        p_hi = attn_1x * p_1x.narrow(1,int(c/2),int(c/2))
        p_lo = scale_as(pred_05x, p_1x)
        joint_pred2 = p_hi + (1 - attn_1x) * p_lo

        joint_pred = self.classifier(torch.cat([joint_pred1, joint_pred2], dim=1))
        pred_05x = self.classifier(pred_05x)
        p_1x = self.classifier(p_1x)
        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(joint_pred, gts)
            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            if cfg.LOSS.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(p_1x, gts, do_rmi=False)
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_lo
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': p_1x,
                'attn_05x': attn_lo,
            }
            return output_dict

    def forward(self, inputs):
        if cfg.MODEL.N_SCALES and not self.training:
            if self.fuse_aspp:
                return self.nscale_fused_forward(inputs, cfg.MODEL.N_SCALES)
            else:
                return self.nscale_forward(inputs, cfg.MODEL.N_SCALES)

        return self.two_scale_forward(inputs)


class MscaleV3Plus(MscaleBase):
    """
    DeepLabV3Plus-based mscale segmentation model
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 use_dpc=False, fuse_aspp=False, attn_2b=False, attn_cls=False, feature_expansion=4):
        super(MscaleV3Plus, self).__init__()
        self.criterion = criterion
        self.fuse_aspp = fuse_aspp
        self.attn_2b = attn_2b
        self.attn_cls = attn_cls
        self.feature_expansion = feature_expansion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)

        # Semantic segmentation prediction head
        bot_ch = cfg.MODEL.SEGATTN_BOT_CH
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, bot_ch, kernel_size=3, padding=1, bias=False),
            Norm2d(bot_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
            Norm2d(bot_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bot_ch, num_classes*self.feature_expansion, kernel_size=1, bias=False),
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(num_classes*self.feature_expansion, num_classes, kernel_size=1, bias=False),
        )

        # Scale-attention prediction head
        if self.attn_2b:
            attn_ch = 2
        else:
            if self.attn_cls:
                attn_ch = num_classes*self.feature_expansion
            else:
                attn_ch = 1

        scale_in_ch = 256 + 48

        self.scale_attn = make_attn_head(in_ch=int(scale_in_ch/2), out_ch=int(attn_ch/2))
        self.scale_attn_rev = make_attn_head(in_ch=int(scale_in_ch/2), out_ch=int(attn_ch/2))

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.bot_fine)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.scale_attn)
            initialize_weights(self.scale_attn_rev)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def _build_scale_tensor(self, scale_float, shape):
        """
        Fill a 2D tensor with a constant scale value
        """
        bs = scale_float.shape[0]
        scale_tensor = None
        for b in range(bs):
            a_tensor = torch.Tensor(1, 1, *shape)
            a_tensor.fill_(scale_float[b])
            if scale_tensor is None:
                scale_tensor = a_tensor
            else:
                scale_tensor = torch.cat([scale_tensor, a_tensor])
        scale_tensor = scale_tensor.cuda()
        return scale_tensor

    def _fwd_feature(self, x):
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)

        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4_attn = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        cat_s4_attn = torch.cat(cat_s4_attn, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])
        return out, aspp, cat_s4_attn

    def _fwd_attn(self, x, cat_s4_attn):
        x_size = x.size()
        c = cat_s4_attn.shape[1]
        scale_attn = self.scale_attn(cat_s4_attn.narrow(1,0,int(c/2)))
        scale_attn = Upsample(scale_attn, x_size[2:])

        if self.attn_2b:
            logit_attn = scale_attn[:, 0:1, :, :]
            aspp_attn = scale_attn[:, 1:, :, :]
        else:
            logit_attn = scale_attn
            aspp_attn = scale_attn

        return logit_attn, aspp_attn

    def _fwd_attn_rev(self, x, cat_s4_attn):
        x_size = x.size()
        c = cat_s4_attn.shape[1]
        scale_attn_rev = self.scale_attn_rev(cat_s4_attn.narrow(1,int(c/2),int(c/2)))
        scale_attn_rev = Upsample(scale_attn_rev, x_size[2:])

        if self.attn_2b:
            logit_attn_rev = scale_attn_rev[:, 0:1, :, :]
            aspp_attn_rev = scale_attn_rev[:, 1:, :, :]
        else:
            logit_attn_rev = scale_attn_rev
            aspp_attn_rev = scale_attn_rev

        return logit_attn_rev, aspp_attn_rev

def DeepV3R50(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='resnet-50', criterion=criterion)


def DeepV3W38(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='wrn38', criterion=criterion)


def DeepV3W38Fuse(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                        fuse_aspp=True)

def DeepV3W38_ATTNCLS(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                        attn_cls=True)

def DeepV3W38Fuse_ATTNCLS(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                        fuse_aspp=True, attn_cls=True)

def DeepV3W38Fuse2(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                        fuse_aspp=True, attn_2b=True)


def DeepV3EffB4(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='efficientnet_b4',
                        criterion=criterion)


def DeepV3EffB4Fuse(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='efficientnet_b4',
                        criterion=criterion, fuse_aspp=True)


def DeepV3X71(num_classes, criterion):
    return MscaleV3Plus(num_classes, trunk='xception71', criterion=criterion)


class MscaleDeeper(MscaleBase):
    """
    Panoptic DeepLab-style semantic segmentation network
    stride8 only
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 fuse_aspp=False, attn_2b=False):
        super(MscaleDeeper, self).__init__()
        self.criterion = criterion
        self.fuse_aspp = fuse_aspp
        self.attn_2b = attn_2b
        self.backbone, s2_ch, s4_ch, high_level_ch = get_trunk(
            trunk_name=trunk, output_stride=8)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256,
                                          output_stride=8)

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.conv_up2 = ConvBnRelu(256 + 64, 256, kernel_size=5, padding=2)
        self.conv_up3 = ConvBnRelu(256 + 32, 256, kernel_size=5, padding=2)
        self.conv_up5 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        # Scale-attention prediction head
        if self.attn_2b:
            attn_ch = 2
        else:
            attn_ch = 1
        self.scale_attn = make_attn_head(in_ch=256,
                                         out_ch=attn_ch)

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.convs2, self.convs4, self.conv_up1,
                               self.conv_up2, self.conv_up3, self.conv_up5,
                               self.scale_attn)

    def _fwd(self, x, aspp_lo=None, aspp_attn=None):
        s2_features, s4_features, final_features = self.backbone(x)
        s2_features = self.convs2(s2_features)
        s4_features = self.convs4(s4_features)
        aspp = self.aspp(final_features)

        if self.fuse_aspp and \
           aspp_lo is not None and aspp_attn is not None:
            aspp_attn = scale_as(aspp_attn, aspp)
            aspp_lo = scale_as(aspp_lo, aspp)
            aspp = aspp_attn * aspp_lo + (1 - aspp_attn) * aspp

        x = self.conv_up1(aspp)
        x = Upsample2(x)
        x = torch.cat([x, s4_features], 1)
        x = self.conv_up2(x)
        x = Upsample2(x)
        x = torch.cat([x, s2_features], 1)
        up3 = self.conv_up3(x)

        out = self.conv_up5(up3)
        out = Upsample2(out)

        scale_attn = self.scale_attn(up3)
        scale_attn = Upsample2(scale_attn)

        if self.attn_2b:
            logit_attn = scale_attn[:, 0:1, :, :]
            aspp_attn = scale_attn[:, 1:, :, :]
        else:
            logit_attn = scale_attn
            aspp_attn = scale_attn

        return out, logit_attn, aspp_attn, aspp


def DeeperW38(num_classes, criterion, s2s4=True):
    return MscaleDeeper(num_classes=num_classes, criterion=criterion,
                        trunk='wrn38')


def DeeperX71(num_classes, criterion, s2s4=True):
    return MscaleDeeper(num_classes=num_classes, criterion=criterion,
                        trunk='xception71')


def DeeperEffB4(num_classes, criterion, s2s4=True):
    return MscaleDeeper(num_classes=num_classes, criterion=criterion,
                        trunk='efficientnet_b4')


class MscaleBasic(MscaleBase):
    """
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(MscaleBasic, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(
            trunk_name=trunk, output_stride=8)

        self.cls_head = make_seg_head(in_ch=high_level_ch,
                                      out_ch=num_classes)
        self.scale_attn = make_attn_head(in_ch=high_level_ch,
                                         out_ch=1)

    def _fwd(self, x, aspp_lo=None, aspp_attn=None, scale_float=None):
        _, _, final_features = self.backbone(x)
        attn = self.scale_attn(final_features)
        pred = self.cls_head(final_features)
        attn = scale_as(attn, x)
        pred = scale_as(pred, x)

        return pred, attn, None, None


def HRNet(num_classes, criterion, s2s4=None):
    return MscaleBasic(num_classes=num_classes, criterion=criterion,
                       trunk='hrnetv2')


class ASPP(MscaleBase):
    """
    ASPP-based Mscale
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(ASPP, self).__init__()
        self.criterion = criterion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=cfg.MODEL.ASPP_BOT_CH,
                                          output_stride=8)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = make_seg_head(in_ch=256, out_ch=num_classes)
        self.scale_attn = make_attn_head(in_ch=256, out_ch=1)

        initialize_weights(self.final)

    def _fwd(self, x, aspp_lo=None, aspp_attn=None, scale_float=None):
        x_size = x.size()
        _, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        aspp = self.bot_aspp(aspp)

        final = self.final(aspp)
        scale_attn = self.scale_attn(aspp)

        out = Upsample(final, x_size[2:])
        scale_attn = Upsample(scale_attn, x_size[2:])

        logit_attn = scale_attn
        aspp_attn = scale_attn

        return out, logit_attn, aspp_attn, aspp


def HRNet_ASP(num_classes, criterion, s2s4=None):
    return ASPP(num_classes=num_classes, criterion=criterion, trunk='hrnetv2')
