"""Official VoxDet LiDAR modules, supervised count heads, no camera pipeline."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .density_head import nonnegative_density
from .voxel_spec import VoxelGridSpec


def lidar_input_features(xyz, spec):
    """Official coarse-cell relative XYZ + absolute XY + normalized range."""
    if not torch.isfinite(xyz).all():
        raise ValueError("Nonfinite LiDAR XYZ")
    if any(n % 2 for n in spec.shape):
        raise ValueError("Official coarse_ratio=2 requires even grid dimensions")
    valid = spec.valid_points(xyz)
    points = xyz[valid].float()
    if len(points) < 2:
        raise ValueError("Too few LiDAR points inside ROI for official Point MLP")
    size = points.new_tensor(spec.size) * 2
    origin = points.new_tensor(spec.origin)
    indices = torch.floor((points.double() - origin.double()) / size.double()).int()
    centers = (indices.float() + .5) * size + origin
    features = torch.cat((points - centers, points[:, :2], points.norm(dim=-1, keepdim=True) / spec.xy_radius), -1)
    return features[None, None], [indices[None]], {"input_points": len(xyz), "roi_points": len(points)}


class VoxDetDensity(nn.Module):
    def __init__(self, spec=None, *, activation_checkpointing=True, output_channels=1, density_bias=-6.):
        super().__init__()
        # Lazy imports let geometry/data tools run without legacy dependencies.
        from .vendor.voxdet.lidar_encoder import LidarEncoder
        from .vendor.voxdet.resnet3d import CustomResNet3D
        from .vendor.voxdet.generalizedfpn import GeneralizedLSSFPN
        from .vendor.voxdet.spatial_deocupled_fpn_dense import SpatiallyDecoupledFPN
        from .vendor.voxdet.det_head import VoxDetHead
        from .vendor.voxdet.occ_head import OccHead
        self.spec = spec or VoxelGridSpec()
        self.activation_checkpointing = activation_checkpointing
        self.output_channels = output_channels
        norm = dict(type="GN", num_groups=32, requires_grad=True)
        fpn = dict(in_channels=[128, 128, 128], out_channels=128, start_level=0,
                   num_outs=3, norm_cfg=norm, conv_cfg=dict(type="Conv3d"),
                   act_cfg=dict(type="ReLU", inplace=True),
                   upsample_cfg=dict(mode="trilinear", align_corners=False))
        self.lidar_tokenizer = LidarEncoder(grid_size=[n // 2 for n in self.spec.shape],
            in_channels=6, out_channels=128, fea_compre=None, base_channels=128,
            split=[8, 8, 8], track_running_stats=True)
        self.voxel_backbone = CustomResNet3D(numC_input=128, num_layer=[2, 2, 2],
            num_channels=[128, 128, 128], stride=[1, 2, 2], drop_path_rate=.3)
        self.voxel_neck = GeneralizedLSSFPN(**fpn)
        # Ident → VoxelAggregatorDual consists solely of these two official
        # modules. Flatten their parameterless registry wrappers, retain all layers.
        self.local_backbone = CustomResNet3D(numC_input=128, num_layer=[3, 3, 3],
            num_channels=[128, 128, 128], stride=[1, 2, 2], norm_cfg=norm, drop_path_rate=.3)
        self.local_neck = SpatiallyDecoupledFPN(share_fpn=False, **fpn)
        head = dict(in_channels=[128], out_channel=output_channels, num_level=1,
            occ_size=list(self.spec.shape), conv_cfg=dict(type="Conv3d", bias=False),
            norm_cfg=norm, balance_cls_weight=False)
        self.main_head = VoxDetHead(num_inst_layer=4, pred_six_directions=True,
            down_sampling_ratio=.5, box_down_sample="trilinear", align_corners=False,
            isolation_scale=0, use_bias=False, balance_reg_loss="none", with_cp=False, **head)
        self.aux_head = OccHead(with_cp=False, **head)
        if output_channels == 1:
            for layer in (self.main_head.cls_convs[0][-1], self.aux_head.occ_convs[0][-1]):
                layer.bias = nn.Parameter(torch.full((1,), density_bias))
        self.register_buffer("valid", self.spec.valid_circle_mask()[None, None], persistent=False)

    def _run(self, module, *args):
        if self.training and self.activation_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, input_xyz):
        features, indices, _ = lidar_input_features(input_xyz, self.spec)
        token = self.lidar_tokenizer(features, indices)
        pyramid = self._run(self.voxel_backbone, token)
        voxel = self._run(self.voxel_neck, pyramid)[0]
        encoded = self._run(self.local_backbone, voxel)
        cls, geometry = self._run(self.local_neck, encoded)
        output = self._run(self.main_head, [[cls[0]], [geometry[0]]])
        density = nonnegative_density(output["output_voxels"], self.valid)
        if density.shape != (1, 1, *self.spec.shape):
            raise ValueError("Official density output shape mismatch")
        if not self.training:
            return {"pred_density": density}
        aux = self._run(self.aux_head, [voxel])
        return {"pred_density": density, "offset": output["output_bbox"],
                "aux_density": nonnegative_density(aux["output_voxels"], self.valid)}
