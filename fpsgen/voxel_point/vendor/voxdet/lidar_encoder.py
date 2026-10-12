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

class LidarEncoder(BaseModule):

    def __init__(self, grid_size, grid_size_rv=None, in_channels=10, out_channels=256, fea_compre=16, base_channels=32, split=[4, 4, 4], track_running_stats=True):
        super(LidarEncoder, self).__init__()
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        self.split = split
        if grid_size_rv is not None:
            self.grid_size_rv = [grid_size[0], grid_size_rv, grid_size[2]]
        self.point_mlp = nn.Sequential(nn.BatchNorm1d(in_channels, track_running_stats=track_running_stats), nn.Linear(in_channels, 64), nn.BatchNorm1d(64, track_running_stats=track_running_stats), nn.ReLU(), nn.Linear(64, 128), nn.BatchNorm1d(128, track_running_stats=track_running_stats), nn.ReLU(), nn.Linear(128, 256), nn.BatchNorm1d(256, track_running_stats=track_running_stats), nn.ReLU(), nn.Linear(256, out_channels))
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(nn.Linear(out_channels, fea_compre), nn.ReLU())

    def save_tpv(self, tpv_list):
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/tpv_tokenizer/pca', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/tpv_tokenizer/pca', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/tpv_tokenizer/pca', 'zx', method='pca')
        pdb.set_trace()
        return

    def forward(self, points, grid_ind):
        (b, n, p, c) = points.shape
        assert b == 1, n == 1
        points = points.view(b * n, p, c)
        device = points[0].get_device()
        (cat_pt_ind, cat_pt_fea) = ([], [])
        for (i_batch, res) in enumerate(grid_ind):
            cat_pt_ind.append(F.pad(grid_ind[i_batch], (1, 0), 'constant', value=i_batch))
        cat_pt_fea = points.squeeze()
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0).squeeze()
        pt_num = cat_pt_ind.shape[0]
        shuffled_ind = torch.randperm(pt_num, device=device)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]
        (unq, unq_inv, unq_cnt) = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        processed_cat_pt_fea = self.point_mlp(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data
        coors = unq.int()
        batch_size = coors[-1][0] + 1
        ret = SparseConvTensor(processed_pooled_data, coors, np.array(self.grid_size), batch_size)
        return ret.dense()
