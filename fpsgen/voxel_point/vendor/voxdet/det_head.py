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

def create_grid(grid_x, grid_y, grid_z, X, Y, Z, device):
    return torch.stack((2 * (grid_z + 0.5) / Z - 1, 2 * (grid_y + 0.5) / Y - 1, 2 * (grid_x + 0.5) / X - 1), dim=-1)

def generate_grid(offsets):
    (B, C, X, Y, Z) = offsets.shape
    device = offsets.device
    grid_x = torch.arange(X, device=device).view(1, X, 1, 1).expand(B, X, Y, Z).float()
    grid_y = torch.arange(Y, device=device).view(1, 1, Y, 1).expand(B, X, Y, Z).float()
    grid_z = torch.arange(Z, device=device).view(1, 1, 1, Z).expand(B, X, Y, Z).float()
    offset_x_pos = offsets[:, 0] * (X - 1)
    offset_x_neg = offsets[:, 1] * (X - 1)
    offset_y_pos = offsets[:, 2] * (Y - 1)
    offset_y_neg = offsets[:, 3] * (Y - 1)
    offset_z_pos = offsets[:, 4] * (Z - 1)
    offset_z_neg = offsets[:, 5] * (Z - 1)
    coords_list = []
    sample_x = torch.clamp(grid_x + offset_x_pos.float(), 0, X - 1)
    coords_list.append(create_grid(sample_x, grid_y, grid_z, X, Y, Z, device))
    sample_x = torch.clamp(grid_x - offset_x_neg.float(), 0, X - 1)
    coords_list.append(create_grid(sample_x, grid_y, grid_z, X, Y, Z, device))
    sample_y = torch.clamp(grid_y + offset_y_pos.float(), 0, Y - 1)
    coords_list.append(create_grid(grid_x, sample_y, grid_z, X, Y, Z, device))
    sample_y = torch.clamp(grid_y - offset_y_neg.float(), 0, Y - 1)
    coords_list.append(create_grid(grid_x, sample_y, grid_z, X, Y, Z, device))
    sample_z = torch.clamp(grid_z + offset_z_pos.float(), 0, Z - 1)
    coords_list.append(create_grid(grid_x, grid_y, sample_z, X, Y, Z, device))
    sample_z = torch.clamp(grid_z - offset_z_neg.float(), 0, Z - 1)
    coords_list.append(create_grid(grid_x, grid_y, sample_z, X, Y, Z, device))
    return coords_list

class VoxelAttentionAggregation(nn.Module):

    def __init__(self, in_channels, num_directions=6, dropout_rate=0.0, use_bias=True, align_corners=False, kernel_size=1, num_groups=32):
        super(VoxelAttentionAggregation, self).__init__()
        self.in_channels = in_channels
        self.num_directions = num_directions
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=use_bias)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=use_bias)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=use_bias)
        self.out_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, bias=use_bias, padding=kernel_size // 2)
        self.query_norm = nn.GroupNorm(num_groups, in_channels)
        self.key_norm = nn.GroupNorm(num_groups, in_channels)
        self.value_norm = nn.GroupNorm(num_groups, in_channels)
        self.out_norm = nn.GroupNorm(num_groups, in_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.align_corners = align_corners

    def forward(self, voxel_feat, coords_list, down_sampling_ratio=0.5):
        (B, C, X, Y, Z) = voxel_feat.shape
        voxel_feat_ds = F.interpolate(voxel_feat, scale_factor=down_sampling_ratio, mode='trilinear', align_corners=self.align_corners)
        (_, _, X_ds, Y_ds, Z_ds) = voxel_feat_ds.shape
        neighbor_feats = []
        for grid in coords_list:
            sampled_feat = F.grid_sample(voxel_feat_ds, grid, mode='bilinear', align_corners=self.align_corners)
            neighbor_feats.append(sampled_feat)
        neighbor_feats = torch.stack(neighbor_feats, dim=1)
        query = self.query_conv(voxel_feat_ds)
        query = self.query_norm(query)
        (B_dir, C, D, H, W) = (B * self.num_directions, C, X_ds, Y_ds, Z_ds)
        neighbor_feats_flat = neighbor_feats.view(B_dir, C, X_ds, Y_ds, Z_ds)
        keys = self.key_conv(neighbor_feats_flat)
        keys = self.key_norm(keys)
        values = self.value_conv(neighbor_feats_flat)
        values = self.value_norm(values)
        keys = keys.view(B, self.num_directions, C, X_ds, Y_ds, Z_ds)
        values = values.view(B, self.num_directions, C, X_ds, Y_ds, Z_ds)
        query_expand = query.unsqueeze(1).expand(-1, self.num_directions, -1, -1, -1, -1)
        attn_scores = (query_expand * keys).sum(dim=2, keepdim=True)
        attn_scores = attn_scores / C ** 0.5
        attn_weights = F.softmax(attn_scores, dim=1)
        aggregated_neighbors = (attn_weights * values).sum(dim=1)
        aggregated_neighbors = self.dropout(aggregated_neighbors)
        aggregated_neighbors_up = F.interpolate(aggregated_neighbors, size=(X, Y, Z), mode='trilinear', align_corners=self.align_corners)
        out = self.out_conv(aggregated_neighbors_up + voxel_feat)
        out = self.out_norm(out)
        return out

class VoxDetHead(nn.Module):

    def __init__(self, in_channels, out_channel, isolation_scale=12, empty_idx=0, num_level=1, with_cp=True, occ_size=[256, 256, 32], loss_weight_cfg=None, balance_cls_weight=True, conv_cfg=dict(type='Conv3d', bias=False), norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), class_frequencies=None, train_cfg=None, test_cfg=None, use_sigmoid=True, reg_loss_type='l1', pred_six_directions=False, scaling_factor=1.0, box_up_sample='trilinear', box_down_sample='trilinear', box_gt_sample='nearest', num_inst_layer=4, ingore_empty=False, dropout_rate=0.0, use_bias=False, balance_reg_loss='none', align_corners=False, kernel_size=1, num_groups=32, down_sampling_ratio=0.5, re_calculate_offset=False, enhance_car=1.0, offset_scale=1.0, reg_loss_weight=1.0, tv_weight=1.0, use_class_mean=False):
        super(VoxDetHead, self).__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.use_class_mean = use_class_mean
        self.out_channel = out_channel
        self.tv_weight = tv_weight
        self.reg_loss_weight = reg_loss_weight
        self.num_level = num_level
        self.empty_idx = empty_idx
        self.use_sigmoid = use_sigmoid
        self.with_cp = with_cp
        self.reg_loss_type = reg_loss_type
        self.isolation_scale = isolation_scale
        self.scaling_factor = scaling_factor
        self.ingore_empty = ingore_empty
        self.balance_reg_loss = balance_reg_loss
        self.box_down_sample = box_down_sample
        self.re_calculate_offset = re_calculate_offset
        self.down_sampling_ratio = down_sampling_ratio
        self.box_gt_sample = box_gt_sample
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {'loss_voxel_ce_weight': 1.0, 'loss_voxel_sem_scal_weight': 1.0, 'loss_voxel_geo_scal_weight': 1.0, 'loss_voxel_ctr_weight': 1.0}
        else:
            self.loss_weight_cfg = loss_weight_cfg
        self.occ_size = occ_size
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_ctr_weight = self.loss_weight_cfg.get('loss_voxel_ctr_weight', 1.0)
        self.train_cfg = train_cfg
        self.box_up_sample = box_up_sample
        self.bbox_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.pred_six_directions = pred_six_directions
        self.refine_conv = nn.ModuleList()
        self.num_inst_layer = num_inst_layer
        for i in range(num_inst_layer):
            self.refine_conv.append(VoxelAttentionAggregation(in_channels=in_channels[0], num_directions=6, dropout_rate=dropout_rate, use_bias=use_bias, align_corners=align_corners, kernel_size=kernel_size, num_groups=num_groups))
        num_box_channels = 3 if not pred_six_directions else 6
        for i in range(self.num_level):
            mid_channel_bbox = self.in_channels[i] // 2
            bbox_conv = nn.Sequential(build_conv_layer(conv_cfg, in_channels=self.in_channels[i], out_channels=mid_channel_bbox, kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, mid_channel_bbox)[1], nn.ReLU(inplace=True), build_conv_layer(conv_cfg, in_channels=mid_channel_bbox, out_channels=num_box_channels, kernel_size=1, stride=1, padding=0))
            self.bbox_convs.append(bbox_conv)
            mid_channel_cls = self.in_channels[i] // 2
            cls_conv = nn.Sequential(build_conv_layer(conv_cfg, in_channels=self.in_channels[i], out_channels=mid_channel_cls, kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, mid_channel_cls)[1], nn.ReLU(inplace=True), build_conv_layer(conv_cfg, in_channels=mid_channel_cls, out_channels=out_channel, kernel_size=1, stride=1, padding=0))
            self.cls_convs.append(cls_conv)
        self.class_frequencies = class_frequencies
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
            if enhance_car > 1.0:
                self.class_weights[1] *= enhance_car
                self.class_weights[4] *= enhance_car
        else:
            self.class_weights = torch.ones(17) / 17
        self.offset_scale = offset_scale

    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None, gt_offset=None):
        cls_feats = [voxel_feats[0][0]]
        box_feats = [voxel_feats[1][0]]
        bbox_outputs = []
        cls_outputs = []
        for (feats, bbox_conv) in zip(box_feats, self.bbox_convs):
            if self.with_cp:
                bbox_outputs.append(torch.utils.checkpoint.checkpoint(bbox_conv, feats))
            else:
                bbox_outputs.append(bbox_conv(feats))
        (_, _, X, Y, Z) = bbox_outputs[0].shape
        output_bbox = F.interpolate(bbox_outputs[0], scale_factor=self.down_sampling_ratio, mode=self.box_down_sample).contiguous()
        output_bbox = output_bbox.sigmoid()
        output_bbox = output_bbox * self.offset_scale
        grid_list = generate_grid(output_bbox)
        for (feats, cls_conv) in zip(cls_feats, self.cls_convs):
            for i in range(self.num_inst_layer):
                feats = self.refine_conv[i](feats, grid_list, self.down_sampling_ratio)
            if self.with_cp:
                cls_outputs.append(torch.utils.checkpoint.checkpoint(cls_conv, feats))
            else:
                cls_outputs.append(cls_conv(feats))
        output_cls = F.interpolate(cls_outputs[0], size=self.occ_size, mode='trilinear', align_corners=False).contiguous()
        result = {'output_bbox': output_bbox, 'output_voxels': output_cls}
        return result
