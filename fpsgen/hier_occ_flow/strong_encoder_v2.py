"""Isolated Stage1 strong-encoder research variants."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .stage1_network import OccupancyConditionEncoder, OccupancyUNet, occupancy_lidar_features, batch_scalar


class FullPyramidVoxDet(OccupancyConditionEncoder):
    """Use the second VoxDet backbone/neck for all three condition scales."""
    def __init__(self, geometry, activation_checkpointing=True):
        super().__init__(geometry, activation_checkpointing, encoder_variant='original')

    def _classification_pyramid(self, inputs):
        neck = self.local_neck
        laterals = list(inputs)
        for i in range(len(laterals)-2, -1, -1):
            up = F.interpolate(laterals[i+1], size=laterals[i].shape[2:], **neck.upsample_cfg)
            x = neck.sp_tpvs_cls[i](laterals[i])
            laterals[i] = neck.fpn_convs_cls[i](neck.lateral_convs_cls[i](torch.cat((x, up), 1)))
        return tuple(laterals)

    def forward(self, lidar_xyz):
        scenes = [lidar_xyz] if isinstance(lidar_xyz, torch.Tensor) else list(lidar_xyz)
        rows = []
        for x in scenes:
            features, indices = occupancy_lidar_features(x, self.spec)
            if getattr(self, '_v6_explicit_fp16', False):
                features = features.to(next(m for m in self.lidar_tokenizer.point_mlp.modules()
                                            if isinstance(m, torch.nn.Linear)).weight.dtype)
            rows.append(self.lidar_tokenizer(features, indices))
        pyramid = self._run(self.voxel_backbone, torch.cat(rows, 0))
        voxel = self._run(self.voxel_neck, pyramid)[0]
        local = self._run(self.local_backbone, voxel)
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            result = checkpoint(self._classification_pyramid, local, use_reentrant=False)
        else:
            result = self._classification_pyramid(local)
        return tuple(self.spec.condition_to_field(x) for x in result)


class FullResolutionConditionFlow(OccupancyUNet):
    """S5: add a zero-initialized 8-channel full-resolution condition residual."""
    def __init__(self, objective, widths, activation_checkpointing):
        super().__init__(objective, widths, activation_checkpointing)
        if widths[0] != 8:
            raise ValueError('S5 requires the registered eight-channel decoder')
        self.fullres_projection = torch.nn.Conv3d(128, 8, 1)
        torch.nn.init.zeros_(self.fullres_projection.weight)
        torch.nn.init.zeros_(self.fullres_projection.bias)

    def prepare_inference_context(self, observed, condition):
        context = super().prepare_inference_context(observed, condition)
        dtype = (torch.get_autocast_gpu_dtype() if observed.device.type == 'cuda'
                 and torch.is_autocast_enabled() else self.stem.weight.dtype)
        context['fullres'] = self.fullres_projection(condition[0].to(dtype))
        return context

    def forward_inference_validated(self, state, observed, condition, time):
        # Preserve S5 semantics; its graph-specific path is not validated yet.
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Validated inference requires eval and no_grad')
        return self.forward(state, observed, condition, time)

    def forward(self, state, observed, condition, time=None, *, inference_context=None):
        if inference_context is not None:
            if inference_context.get('observed') is not observed:
                raise ValueError('Inference context does not match observed mask')
            projected = inference_context['condition_projected']
            fullres = inference_context['fullres']
        else:
            if condition is None or len(condition) != 3:
                raise ValueError('Three conditions required')
            projected = None
            fullres = None
        if state.ndim != 5 or state.shape[1] != 1 or observed.shape != state.shape:
            raise ValueError('Scalar state and matching observed mask required')
        if self.objective != 'fm' or time is None:
            raise ValueError('S5 is registered only for FM')
        t = batch_scalar(time, state)
        if not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError('t outside [0,1]')
        angle = t[:, None] * self.frequencies[None]
        embedding = self.embedding(torch.cat((angle.sin(), angle.cos()), 1).to(self.embedding[0].weight.dtype))
        x = self.stem(torch.cat((state, observed.float()), 1).to(self.stem.weight.dtype))
        skips = []
        for i in range(4):
            if i:
                x = self.down[i-1](x)
                c = self.condition[i-1](condition[i-1].to(x.dtype)) if projected is None else projected[i-1]
                if c.shape != x.shape:
                    raise ValueError('VoxDet condition ZYX pyramid mismatch')
                x = x + c
            x = self._block(self.enc[i], x, embedding)
            skips.append(x)
        for i in reversed(range(3)):
            x = self.up[i](F.interpolate(x, size=skips[i].shape[-3:], mode='trilinear', align_corners=False)) + skips[i]
            x = self._block(self.dec[i], x, embedding)
        if fullres is None:
            fullres = self.fullres_projection(condition[0].to(x.dtype))
        x = x + F.interpolate(fullres, size=x.shape[-3:], mode='trilinear', align_corners=False)
        return self.output(x).float()
