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

class BasicBlock3D(nn.Module):

    def __init__(self, channels_in, channels_out, stride=1, downsample=None, drop_path_rate=0.0):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU', inplace=True))
        self.conv2 = ConvModule(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop_path(out)
        out = out + identity
        return self.relu(out)

class CustomResNet3D(nn.Module):

    def __init__(self, numC_input, num_layer=[2, 2, 2], num_channels=None, stride=[2, 2, 2], drop_path_rate=0.0, backbone_output_ids=None, with_cp=False, norm_cfg=dict(type='BN3d')):
        super(CustomResNet3D, self).__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        total_blocks = sum(num_layer)
        block_idx = 0
        for i in range(len(num_layer)):
            dpr = drop_path_rate * block_idx / total_blocks
            block_idx += 1
            layer = [BasicBlock3D(curr_numC, num_channels[i], stride=stride[i], downsample=ConvModule(curr_numC, num_channels[i], kernel_size=3, stride=stride[i], padding=1, bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=norm_cfg, act_cfg=None), drop_path_rate=dpr)]
            curr_numC = num_channels[i]
            for _ in range(num_layer[i] - 1):
                dpr = drop_path_rate * block_idx / total_blocks
                block_idx += 1
                layer.append(BasicBlock3D(curr_numC, curr_numC, drop_path_rate=dpr))
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for (lid, layer) in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
