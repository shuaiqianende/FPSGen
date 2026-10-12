# Generated from the pinned official VoxDet source; see SOURCE_MANIFEST.json.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch_scatter
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from timm.layers import DropPath
from torchvision.ops import DeformConv2d

class OccHead(nn.Module):

    def __init__(self, in_channels, out_channel, empty_idx=0, num_level=1, with_cp=True, occ_size=[256, 256, 32], loss_weight_cfg=None, balance_cls_weight=True, conv_cfg=dict(type='Conv3d', bias=False), norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), class_frequencies=None, train_cfg=None, test_cfg=None):
        super(OccHead, self).__init__()
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.empty_idx = empty_idx
        self.with_cp = with_cp
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {'loss_voxel_ce_weight': 1.0, 'loss_voxel_sem_scal_weight': 1.0, 'loss_voxel_geo_scal_weight': 1.0}
        else:
            self.loss_weight_cfg = loss_weight_cfg
        self.occ_size = occ_size
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.train_cfg = train_cfg
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(build_conv_layer(conv_cfg, in_channels=self.in_channels[i], out_channels=mid_channel, kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, mid_channel)[1], nn.ReLU(inplace=True), build_conv_layer(conv_cfg, in_channels=mid_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0))
            self.occ_convs.append(occ_conv)
        self.class_frequencies = class_frequencies
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17) / 17

    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        output_occs = []
        for (feats, occ_conv) in zip(voxel_feats, self.occ_convs):
            if self.with_cp:
                output_occs.append(torch.utils.checkpoint.checkpoint(occ_conv, feats))
            else:
                output_occs.append(occ_conv(feats))
        result = {'output_voxels': F.interpolate(output_occs[0], size=self.occ_size, mode='trilinear', align_corners=False).contiguous()}
        return result
