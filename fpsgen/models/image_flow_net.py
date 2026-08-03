
"""Network components for the Flexible Condition BEV Flow Prior.

``BEVFlowTransNet`` combines a Mini-PointPillar LiDAR encoder, a multi-scale
layout-condition pyramid, a convolutional U-Net, and an AdaLN-modulated
transformer bottleneck to parameterize the BEV velocity :math:`v_\phi`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import pykeops

@torch.no_grad()
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    pykeops.set_verbose(False)
    xi = pykeops.torch.LazyTensor(q_points.unsqueeze(-2))
    xj = pykeops.torch.LazyTensor(s_points.unsqueeze(-3))
    dij = (xi - xj).sqnorm2()
    knn_d2, knn_indices = dij.Kmin_argKmin(k, dim=q_points.dim() - 1)
    return knn_d2, knn_indices

class DynamicKNNPillarEncoder(nn.Module):
    """Mini-PointPillar encoder for the LiDAR condition :math:`c_l`.

    Local kNN edge features augment metric coordinates, per-pillar centroid
    offsets, and cell-center offsets. Point features are max-pooled into the
    dense ``[x_index, y_index]`` BEV convention used throughout FPSGen.
    """
    def __init__(self, in_channels=3, out_channels=64, grid_size=256, pc_range=50.0, k=16):
        super().__init__()
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = (pc_range * 2.0) / grid_size
        self.k = k

        edge_in_channels = in_channels * 2
        self.knn_channels = 32
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_channels, self.knn_channels, bias=False),
            nn.LayerNorm(self.knn_channels),
            nn.ReLU(inplace=True)
        )

        self.augmented_dim = 8 + self.knn_channels

        self.pfn = nn.Sequential(
            nn.Linear(self.augmented_dim, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True)
        )

        self.out_channels = out_channels

    def get_edge_features(self, points: Tensor, knn_indices: Tensor) -> Tensor:
        """Construct center-relative edge features for local geometric aggregation."""
        B, N, C = points.shape
        _, _, k = knn_indices.shape
        batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1, 1).expand(B, N, k)
        knn_points = points[batch_indices, knn_indices.long(), :]
        center_points = points.unsqueeze(2).expand(B, N, k, C)
        return torch.cat([knn_points - center_points, center_points], dim=-1)

    def forward(self, points):
        """Pillarize batched XYZ coordinates into a dense condition map.

        Args:
            points: Metric XYZ tensor with shape ``[B, N, 3]``.

        Returns:
            Dense features with shape ``[B, out_channels, G, G]``.
        """
        B, N, C = points.shape
        device = points.device

        _, knn_indices = keops_knn(points, points, self.k)
        edge_features = self.get_edge_features(points, knn_indices)

        edge_features_flat = edge_features.view(B * N * self.k, -1)
        local_features = self.edge_mlp(edge_features_flat)
        local_features = local_features.view(B, N, self.k, -1)
        local_features, _ = torch.max(local_features, dim=2)

        xy = points[..., :2]
        xy_norm = (xy + self.pc_range) / (self.pc_range * 2.0)
        xy_pixels = (xy_norm * self.grid_size).long()

        valid_mask = (xy_pixels[..., 0] >= 0) & (xy_pixels[..., 0] < self.grid_size) & \
                     (xy_pixels[..., 1] >= 0) & (xy_pixels[..., 1] < self.grid_size)

        num_pixels = self.grid_size * self.grid_size
        flat_indices = xy_pixels[..., 0] * self.grid_size + xy_pixels[..., 1]
        flat_indices = torch.where(valid_mask, flat_indices, num_pixels)

        sum_features = torch.zeros((B, num_pixels + 1, 3), device=device)
        counts = torch.zeros((B, num_pixels + 1, 1), device=device)

        indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, 3)
        sum_features.scatter_add_(1, indices_expanded, points)
        counts.scatter_add_(1, flat_indices.unsqueeze(-1), torch.ones_like(points[..., :1]))

        mean_features = sum_features / (counts + 1e-5)

        point_means = torch.gather(mean_features, 1, indices_expanded)
        offset_c = points - point_means

        grid_center_x = xy_pixels[..., 0:1] * self.voxel_size - self.pc_range + self.voxel_size / 2.0
        grid_center_y = xy_pixels[..., 1:2] * self.voxel_size - self.pc_range + self.voxel_size / 2.0
        offset_p = xy - torch.cat([grid_center_x, grid_center_y], dim=-1)

        augmented_points = torch.cat([points, offset_c, offset_p, local_features], dim=-1)

        point_features = augmented_points.view(B * N, -1)
        point_features = self.pfn(point_features)
        point_features = point_features.view(B, N, self.out_channels)

        bev_features = torch.full((B, self.out_channels, num_pixels + 1), -1e9, device=device)
        flat_indices_expanded = flat_indices.unsqueeze(1).expand(-1, self.out_channels, -1)
        point_features = point_features.permute(0, 2, 1).contiguous()

        bev_features.scatter_reduce_(2, flat_indices_expanded, point_features, reduce="amax", include_self=False)

        bev_features = torch.where(bev_features == -1e9, torch.tensor(0.0, device=device), bev_features)

        bev_features = bev_features[:, :, :-1]
        bev_features = bev_features.view(B, self.out_channels, self.grid_size, self.grid_size)

        return bev_features

class DepthPillarEncoder(nn.Module):
    """Optional camera-depth encoder retained for input-modality extensions."""
    def __init__(self, in_channels=1, feature_channels=32, grid_size=256, pc_range=50.0):
        super().__init__()
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.feature_channels = feature_channels

        self.register_buffer('fx', torch.tensor(718.856))
        self.register_buffer('fy', torch.tensor(718.856))
        self.register_buffer('cx', torch.tensor(607.1928))
        self.register_buffer('cy', torch.tensor(185.2157))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True)
        )

    def backproject_to_3d(self, depth_map: Tensor) -> Tensor:
        """Back-project depth using the registered camera intrinsics."""
        B, _, H, W = depth_map.shape
        device = depth_map.device

        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        u = u.expand(B, H, W)
        v = v.expand(B, H, W)

        z_cam = depth_map.squeeze(1)
        x_cam = (u - self.cx) * z_cam / self.fx
        y_cam = (v - self.cy) * z_cam / self.fy

        return torch.stack([x_cam, y_cam, z_cam], dim=-1)

    def forward(self, depth_map: Tensor):
        """Encode ``[B,1,H,W]`` depth into ``[B,C,G,G]`` BEV features.

        Camera coordinates are converted to ego-frame XY before max-pooling
        visible depth features into pillars.
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device

        features_2d = self.feature_extractor(depth_map)

        cam_points = self.backproject_to_3d(depth_map)

        cam_points = cam_points.view(B, -1, 3)
        features = features_2d.view(B, self.feature_channels, -1)

        ego_x = cam_points[..., 2]
        ego_y = -cam_points[..., 0]
        ego_xy = torch.stack([ego_x, ego_y], dim=-1)

        xy_norm = (ego_xy + self.pc_range) / (self.pc_range * 2.0)
        xy_pixels = (xy_norm * self.grid_size).long()

        valid_mask = (xy_pixels[..., 0] >= 0) & (xy_pixels[..., 0] < self.grid_size) & \
                     (xy_pixels[..., 1] >= 0) & (xy_pixels[..., 1] < self.grid_size) & \
                     (cam_points[..., 2] > 0)

        num_pixels = self.grid_size * self.grid_size
        flat_indices = xy_pixels[..., 0] * self.grid_size + xy_pixels[..., 1]

        flat_indices = torch.where(valid_mask, flat_indices, num_pixels)

        bev_features = torch.full((B, self.feature_channels, num_pixels + 1), -1e9, device=device)
        flat_indices_expanded = flat_indices.unsqueeze(1).expand(-1, self.feature_channels, -1)

        bev_features.scatter_reduce_(2, flat_indices_expanded, features, reduce="amax", include_self=False)

        bev_features = torch.where(bev_features == -1e9, torch.tensor(0.0, device=device), bev_features)

        bev_features = bev_features[:, :, :-1]
        bev_features = bev_features.view(B, self.feature_channels, self.grid_size, self.grid_size)

        return bev_features

class MultiScaleConditionEncoder(nn.Module):
    """Build the condition pyramid consumed at all four U-Net resolutions."""

    def __init__(self, in_channels=4, base_ch=64):
        super().__init__()

        self.level_0 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1)
        )

        self.level_1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        )

        self.level_2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(16, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1)
        )

        self.level_3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.GroupNorm(16, base_ch * 8),
            nn.SiLU(),
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1)
        )

    def forward(self, x):
        """Return a feature pyramid from finest to coarsest spatial resolution."""
        c0 = self.level_0(x)
        c1 = self.level_1(c0)
        c2 = self.level_2(c1)
        c3 = self.level_3(c2)

        return [c0, c1, c2, c3]

class SpatialConditionFusion(nn.Module):
    """Fuse same-resolution state and condition features by channel projection."""

    def __init__(self, feature_ch, cond_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(feature_ch + cond_ch, feature_ch, kernel_size=1),
            nn.SiLU()
        )

    def forward(self, h, c):
        return self.proj(torch.cat([h, c], dim=1))

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(self.norm(x))
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N)

        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        h = attn @ v
        h = h.transpose(-2, -1)
        h = h.reshape(B, C, H, W)

        return x + self.proj(h)

class ResBlock(nn.Module):
    """Residual block modulated by continuous Flow-Matching time."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))

        t_hidden = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_hidden

        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)

class DiTBlock2D_SpatialAdaLN(nn.Module):
    """AdaLN transformer bottleneck with spatial cross-attention conditions."""
    def __init__(self, hidden_size, time_dim, cond_dim, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            kdim=cond_dim,
            vdim=cond_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb, cond):
        B, C, H, W = x.shape

        x_seq = x.view(B, C, -1).permute(0, 2, 1)

        cond_seq = cond.view(B, cond.shape[1], -1).permute(0, 2, 1)

        mod_params = self.adaLN_modulation(t_emb).unsqueeze(1)
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = mod_params.chunk(6, dim=-1)

        norm1_x = self.norm1(x_seq) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(norm1_x, norm1_x, norm1_x, need_weights=False)
        x_seq = x_seq + gate_msa * attn_out

        norm_cross_x = self.norm_cross(x_seq)
        cross_out, _ = self.cross_attn(query=norm_cross_x, key=cond_seq, value=cond_seq, need_weights=False)
        x_seq = x_seq + cross_out

        norm2_x = self.norm2(x_seq) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(norm2_x)
        x_seq = x_seq + gate_mlp * mlp_out

        out = x_seq.permute(0, 2, 1).view(B, C, H, W)

        return out

class DiTBlock2D(nn.Module):
    """Alternative global-condition AdaLN transformer block.

    This component is retained for architecture ablations; the released
    ``BEVFlowTransNet`` uses ``DiTBlock2D_SpatialAdaLN``.
    """

    def __init__(self, hidden_size, time_dim, cond_dim, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim + cond_dim, 9 * hidden_size, bias=True)
        )

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb, cond):
        """Apply self-attention, condition cross-attention, and an MLP.

        Args:
            x: State features ``[B, C, H, W]``.
            t_emb: Continuous-time embedding ``[B, time_dim]``.
            cond: Co-located condition features ``[B, cond_dim, H, W]``.
        """
        B, C, H, W = x.shape
        cond_dim = cond.shape[1]

        global_cond = self.global_pool(cond).view(B, -1)
        t_cond_emb = torch.cat([t_emb, global_cond], dim=1)

        x_seq = x.view(B, C, -1).permute(0, 2, 1)
        cond_seq = cond.view(B, cond_dim, -1).permute(0, 2, 1)

        mod_params = self.adaLN_modulation(t_cond_emb).chunk(9, dim=1)
        (shift_msa, scale_msa, gate_msa,
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = [p.unsqueeze(1) for p in mod_params]

        norm1_x = self.norm1(x_seq) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(norm1_x, norm1_x, norm1_x, need_weights=False)
        x_seq = x_seq + gate_msa * attn_out

        norm_cross_x = self.norm_cross(x_seq) * (1 + scale_cross) + shift_cross
        cross_out, _ = self.cross_attn(query=norm_cross_x, key=cond_seq, value=cond_seq, need_weights=False)
        x_seq = x_seq + gate_cross * cross_out

        norm2_x = self.norm2(x_seq) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(norm2_x)
        x_seq = x_seq + gate_mlp * mlp_out

        out = x_seq.permute(0, 2, 1).view(B, C, H, W)
        return out

class BEVFlowTransNet(nn.Module):
    """Parameterize the conditional BEV velocity field :math:`v_\phi`.

    The active tuple :math:`C_m` is encoded from a LiDAR pillar map and two
    semantic layout channels. Multi-scale fusion preserves local structure,
    while the transformer bottleneck supplies global scene interactions.
    """
    def __init__(self, base_ch=32, time_dim=512, cls=20, layout_ch=2):
        super().__init__()
        self.cls = cls
        self.time_dim = time_dim

        self.pc_encoder = DynamicKNNPillarEncoder(
            in_channels=3,
            out_channels=base_ch,
            grid_size=256,
            pc_range=50.0,
            k=16
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(256, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        self.cond_fpn = MultiScaleConditionEncoder(in_channels=base_ch + layout_ch, base_ch=base_ch)

        self.init_conv = nn.Conv2d(3+cls, base_ch, kernel_size=3, padding=1)

        dims = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        cond_dims = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.downs = nn.ModuleList()
        in_ch = base_ch
        for level, out_ch in enumerate(dims):
            layer = nn.ModuleDict()
            layer["cond_fuse"] = SpatialConditionFusion(in_ch, cond_dims[level])
            layer["res1"] = ResBlock(in_ch, out_ch, time_dim)
            layer["res2"] = ResBlock(out_ch, out_ch, time_dim)
            if level != len(dims) - 1:
                layer["downsample"] = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            self.downs.append(layer)
            in_ch = out_ch

        self.pos_embed = nn.Parameter(torch.zeros(1, base_ch * 8, 32, 32))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        mid_dim = dims[-1]
        self.mid_fuse = SpatialConditionFusion(mid_dim, cond_dims[-1])
        num_dit_blocks = 8
        self.mid_blocks = nn.ModuleList([
            DiTBlock2D_SpatialAdaLN(hidden_size=mid_dim, time_dim=time_dim, cond_dim=base_ch * 8, num_heads=8)
            for _ in range(num_dit_blocks)
        ])

        self.ups = nn.ModuleList()
        in_ch = dims[-1]
        for level, out_ch in reversed(list(enumerate(dims))):
            layer = nn.ModuleDict()
            layer["cond_fuse"] = SpatialConditionFusion(in_ch + out_ch, cond_dims[level])
            layer["res1"] = ResBlock(in_ch + out_ch, out_ch, time_dim)
            layer["res2"] = ResBlock(out_ch + out_ch, out_ch, time_dim)
            if level != 0:
                layer["upsample"] = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1)
                )
            self.ups.append(layer)
            in_ch = out_ch

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, dims[0]), nn.SiLU(), nn.Conv2d(dims[0], 3+cls, 3, padding=1)
        )

        nn.init.zeros_(self.final_conv[-1].weight)
        nn.init.zeros_(self.final_conv[-1].bias)

    def get_raw_pc_bev(self, points):
        """Encode input points into the LiDAR conditioning BEV feature map."""
        return self.pc_encoder(points)

    def get_time_embedding(self, t, dim=256):
        """Create a sinusoidal embedding for continuous BEV flow times in [0, 1]."""
        t_scaled = t * 1000.0
        half_dim = dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        emb = t_scaled[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, xt, t, raw_pc, layout_mask):
        """Predict :math:`v_\phi(B_\tau,\tau,C_m)`.

        Args:
            xt: Current BEV path state ``B_tau`` with shape ``[B,3,G,G]``.
            t: Flow time ``tau`` with shape ``[B]``.
            raw_pc: Mini-PointPillar LiDAR features.
            layout_mask: Vehicle and road/ground condition maps.
        """
        t_emb = self.time_mlp(self.get_time_embedding(t, dim=256))
        raw_cond = torch.cat([raw_pc, layout_mask], dim=1)

        c_fpn = self.cond_fpn(raw_cond)

        h = self.init_conv(xt)
        skips = []

        for level, layer_dict in enumerate(self.downs):
            h = layer_dict["cond_fuse"](h, c_fpn[level])
            h = layer_dict["res1"](h, t_emb)
            skips.append(h)

            h = layer_dict["res2"](h, t_emb)
            skips.append(h)

            if "attn" in layer_dict:
                h = layer_dict["attn"](h)
            if "downsample" in layer_dict:
                h = layer_dict["downsample"](h)

        h = self.mid_fuse(h, c_fpn[-1])
        h = h + self.pos_embed
        for dit_block in self.mid_blocks:
            h = dit_block(h, t_emb, c_fpn[-1])

        reversed_levels = list(reversed(range(len(self.downs))))

        for idx, layer_dict in enumerate(self.ups):
            current_level = reversed_levels[idx]

            h = torch.cat([h, skips.pop()], dim=1)
            h = layer_dict["cond_fuse"](h, c_fpn[current_level])
            h = layer_dict["res1"](h, t_emb)

            h = torch.cat([h, skips.pop()], dim=1)
            h = layer_dict["res2"](h, t_emb)

            if "attn" in layer_dict:
                h = layer_dict["attn"](h)
            if "upsample" in layer_dict:
                h = layer_dict["upsample"](h)

        v_pred = self.final_conv(h)
        return v_pred
