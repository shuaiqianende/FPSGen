"""384-grid direct surface-hit occupancy objectives over the existing VoxDet+3D UNet.

This is a *new* occupancy objective. It must never load historical count-mapping
weights as an occupancy head or interpret Poisson counts as raw-map hits.
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def ids_to_dense_occupancy(parent_ids, geometry, *, device=None):
    """Convert sorted sparse X-fastest parent IDs to [1,1,Z,Y,X] binary targets."""
    # PyTorch2.0 cannot ingest NumPy uint32, even with dtype=torch.long.
    if isinstance(parent_ids,np.ndarray): parent_ids=parent_ids.astype(np.int64)
    ids = torch.as_tensor(parent_ids, device=device, dtype=torch.long)
    if ids.ndim != 1 or not ids.numel():
        raise ValueError("Expected nonempty one-dimensional parent IDs")
    if bool(((ids < 0) | (ids >= geometry.num_parents)).any()):
        raise ValueError("Parent ID outside configured coarse grid")
    if len(ids) > 1 and bool((ids[1:] <= ids[:-1]).any()):
        raise ValueError("Parent IDs must be strictly increasing and unique")
    nx, ny, nz = geometry.parent_resolution
    field = torch.zeros((1, 1, nz, ny, nx), device=ids.device, dtype=torch.float32)
    field.view(-1)[ids] = 1.0
    return field


def occupancy_flow_pair(occupied, valid_mask, *, generator, time=None):
    """Gaussian -> symmetric binary occupation; uniform scene-shared t."""
    if generator is None:
        raise ValueError("Explicit RNG is required")
    if occupied.ndim != 5 or occupied.shape[1:] != valid_mask.shape[1:] or valid_mask.shape[0] not in (1,occupied.shape[0]):
        raise ValueError("Dense occupancy and ROI mask shape mismatch")
    if bool(((occupied != 0) & (occupied != 1)).any()):
        raise ValueError("Occupancy must be binary")
    valid = valid_mask.bool().expand_as(occupied)
    if not bool(valid.any()):
        raise ValueError("No valid voxels")
    if bool((occupied.bool() & ~valid).any()):
        raise ValueError("GT contains occupied voxels outside representation ROI")
    x1 = (2.0 * occupied.float() - 1.0) * valid
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator) * valid
    if time is None:
        time = torch.rand((len(x1),), device=x1.device, generator=generator)
    if time.numel() != len(x1) or not bool(((time >= 0) & (time <= 1)).all()):
        raise ValueError("One independent scene-shared time per batch item is required")
    time = time.reshape(len(x1));weight=time.reshape(-1,1,1,1,1)
    return {"state": (1 - weight) * x0 + weight * x1,
            "velocity": x1 - x0, "source": x0, "target": x1, "time": time}


def masked_occ_loss(prediction, target, valid_mask, *, objective):
    if prediction.shape != target.shape or target.shape[1:] != valid_mask.shape[1:] or valid_mask.shape[0] not in (1,target.shape[0]):
        raise ValueError("Prediction/target/ROI mask mismatch")
    valid = valid_mask.bool().expand_as(target)
    if objective == "fm":
        per_voxel = (prediction.float() - target.float()).square()
    elif objective == "bce":
        per_voxel = F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none")
    else:
        raise ValueError("Expected fm or bce objective")
    return (per_voxel * valid).sum() / valid.sum().clamp_min(1)


class Stage1DirectOccupancy(nn.Module):
    """Trainable LiDAR encoder plus full-resolution occupancy FM or BCE.

    No GT ever reaches encode_condition or the model's forward velocity.
    The Gaussian bridge is used ONLY for the FM objective, not for direct BCE.
    """

    def __init__(self, geometry, *, objective="fm", widths=(8, 16, 32, 64),
                 activation_checkpointing=True, encoder_variant="original",
                 omd_config=None, dense_fidelity_config=None):
        super().__init__()
        if objective not in {"fm", "bce"}:
            raise ValueError("objective must be fm or bce")
        from .stage1_network import OccupancyGridSpec,OccupancyConditionEncoder,OccupancyUNet
        self.objective = objective
        self.hierarchy_geometry = geometry
        nx, ny, nz = geometry.parent_resolution
        if tuple(geometry.parent_resolution)!=(384,384,32) or tuple(geometry.child_split)!=(2,2,2):
            raise ValueError('This independent Stage1 is fixed to384+8way')
        self.geometry = OccupancyGridSpec(
            x_min=geometry.origin[0], y_min=geometry.origin[1], z_min=geometry.origin[2],
            x_max=geometry.maximum[0], y_max=geometry.maximum[1], z_max=geometry.maximum[2],
            nx=nx, ny=ny, nz=nz, xy_radius=geometry.sphere_radius,
            sphere_radius=geometry.sphere_radius)
        self.encoder=OccupancyConditionEncoder(
            self.geometry,activation_checkpointing,
            encoder_variant=encoder_variant,omd_config=omd_config,
            dense_fidelity_config=dense_fidelity_config)
        self.flow=OccupancyUNet(objective,widths,activation_checkpointing)
        self.register_buffer('valid',self.geometry.field_mask(),persistent=False)
        self.last_condition_cache_cuda_events=None

    def encode_condition(self, lidar_xyz):
        return self.encoder(lidar_xyz)

    def forward(self, lidar_xyz, *, state=None, time=None, condition=None):
        if condition is None:
            condition = self.encode_condition(lidar_xyz)
        scenes=[lidar_xyz] if isinstance(lidar_xyz,torch.Tensor) else list(lidar_xyz)
        observed = torch.cat([self.geometry.observed_mask(x) for x in scenes],0)
        if self.objective == "fm":
            if state is None or time is None:
                raise ValueError("FM requires state and time")
            prediction = self.flow(state,observed,condition,time)
        else:
            if state is not None or time is not None:
                raise ValueError("BCE is deterministic and does not accept x_t/t")
            prediction = self.flow(torch.zeros_like(observed,dtype=torch.float32),observed,condition)
        return prediction*self.valid

    @torch.no_grad()
    def sample(self, lidar_xyz, *, generator=None, nfe=1, condition=None,
               condition_cache=False):
        if condition is None:
            condition = self.encode_condition(lidar_xyz)
        if self.objective == "bce":
            logits = self(lidar_xyz, condition=condition)
            return logits.sigmoid() * self.valid, logits
        if nfe not in (1, 2, 4, 8, 16) or generator is None:
            raise ValueError("FM sampling requires allowed NFE and seeded generator")
        source = torch.randn(
            (1, 1, *self.geometry.field_shape), device=lidar_xyz.device,
            dtype=torch.float32, generator=generator) * self.valid
        observed = self.geometry.observed_mask(lidar_xyz)
        self.last_condition_cache_cuda_events = None
        if condition_cache:
            if observed.device.type == "cuda":
                cache_start, cache_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                cache_start.record()
                inference_context = self.flow.prepare_inference_context(observed, condition)
                cache_end.record()
                self.last_condition_cache_cuda_events = (cache_start, cache_end)
            else:
                inference_context = self.flow.prepare_inference_context(observed, condition)
        else:
            inference_context = None
        current = source
        for step in range(nfe):
            if inference_context is None:
                v=self.flow(current,observed,condition,current.new_tensor(float(step)/nfe))
            else:
                v=self.flow(current,observed,None,current.new_tensor(float(step)/nfe),
                            inference_context=inference_context)
            v=v*self.valid
            current = current + v.float() / nfe
        # Uncalibrated occupancy score. Final threshold must be chosen on development.
        score = ((current.float() + 1) * 0.5).clamp(0, 1) * self.valid
        return score, current
