"""PWM adapter around the vendored official PointTransformerV3 class.

The PTv3 trunk itself is imported from ``ptv3_reference.model``. This file
only converts PWM's variable-length point tensors into the official Point
dictionary and maps decoded token features back to the original points.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointTransformerV3
from fpsgen.voxel_point.vendor.pwm.ptv3_time_conditioning import (
    build_official_ptv3_with_time_context,
)


def _segment_mean(values, indices, count):
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, indices, values)
    # The supported CUDA/PyTorch combination has no deterministic bincount
    # kernel.  This is bookkeeping only; keep deterministic mode for the
    # network while temporarily allowing this one operation.
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    except AttributeError:
        warn_only = False
    if deterministic:
        torch.use_deterministic_algorithms(False)
    try:
        counts = torch.bincount(indices, minlength=count)
    finally:
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
    counts = counts.to(values.dtype).clamp_min_(1)
    return output / counts.unsqueeze(1)


class OfficialPTV3PointFlowNet(nn.Module):
    """Official PTv3 hierarchy with PWM's flow-velocity interface."""

    def __init__(
        self, in_channels=6, base_channels=32, quantization_size=0.02,
        coordinate_scale=(1.0, 1.0, 1.0), fine_token_size=0.05,
        coarse_token_size=0.10, flow_voxel_size=None,
        intra_voxel_encoding_bands=4, fine_dim=None, coarse_dim=None,
        fine_depth=3, coarse_depth=4, decoder_depth=2, fine_heads=8,
        coarse_heads=8, fine_window_size=64, coarse_window_size=128,
        dropout=0.0, point_norm="batch",
        serialization_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders=True, mlp_ratio=4.0, drop_path=0.3,
        pre_norm=True, qkv_bias=True, enable_rpe=False, enable_flash=True,
        upcast_attention=False, upcast_softmax=False,
        enc_depths: Optional[Sequence[int]] = None,
        enc_channels: Optional[Sequence[int]] = None,
        enc_heads: Optional[Sequence[int]] = None,
        enc_patch_sizes: Optional[Sequence[int]] = None,
        dec_depths: Optional[Sequence[int]] = None,
        dec_channels: Optional[Sequence[int]] = None,
        dec_heads: Optional[Sequence[int]] = None,
        dec_patch_sizes: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if quantization_size <= 0 or fine_token_size <= 0 or coarse_token_size < fine_token_size:
            raise ValueError("invalid PTv3 point/token sizes")
        fine_dim = int(fine_dim or base_channels * 4)
        coarse_dim = int(coarse_dim or base_channels * 8)
        if fine_dim % fine_heads or coarse_dim % coarse_heads:
            raise ValueError("PTv3 channels must be divisible by attention heads")
        if isinstance(serialization_order, str):
            serialization_order = (serialization_order,)
        else:
            serialization_order = tuple(serialization_order)
        if not serialization_order:
            raise ValueError("serialization_order must contain at least one order")
        if enable_flash and (enable_rpe or upcast_attention or upcast_softmax):
            raise ValueError(
                "FlashAttention requires enable_rpe=false, "
                "upcast_attention=false, and upcast_softmax=false"
            )
        self.input_channels = int(in_channels)
        self.quantization_size = float(quantization_size)
        self.fine_token_size = float(fine_token_size)
        self.coarse_token_size = float(coarse_token_size)
        self.intra_voxel_encoding_bands = int(intra_voxel_encoding_bands)
        self.register_buffer("coordinate_scale", torch.as_tensor(coordinate_scale, dtype=torch.float32), persistent=False)
        if flow_voxel_size is None:
            flow_voxel_size = (fine_token_size,) * 3
        self.register_buffer("flow_voxel_size", torch.as_tensor(flow_voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "intra_voxel_frequencies",
            2.0 ** torch.arange(self.intra_voxel_encoding_bands, dtype=torch.float32) * math.pi,
            persistent=False,
        )
        self.time_channels = 32
        self.register_buffer(
            "time_frequencies",
            torch.exp(-math.log(10000.0) * torch.arange(self.time_channels // 2, dtype=torch.float32)
                      / max(self.time_channels // 2 - 1, 1)),
            persistent=False,
        )
        # Time is injected inside every official PTv3 block through its
        # scene-level ``context`` field. Keep it out of the token input so
        # the backbone receives only spatial/conditioning features here.
        official_input_channels = in_channels + 6 * self.intra_voxel_encoding_bands
        architecture = {
            "enc_depths": enc_depths,
            "enc_channels": enc_channels,
            "enc_heads": enc_heads,
            "enc_patch_sizes": enc_patch_sizes,
            "dec_depths": dec_depths,
            "dec_channels": dec_channels,
            "dec_heads": dec_heads,
            "dec_patch_sizes": dec_patch_sizes,
        }
        specified = [name for name, value in architecture.items() if value is not None]
        if specified and len(specified) != len(architecture):
            missing = sorted(set(architecture) - set(specified))
            raise ValueError(
                "PTv3 full architecture must specify all encoder/decoder lists "
                f"together; missing: {', '.join(missing)}"
            )

        if specified:
            enc_depths = tuple(int(value) for value in enc_depths)
            enc_channels = tuple(int(value) for value in enc_channels)
            enc_heads = tuple(int(value) for value in enc_heads)
            enc_patches = tuple(int(value) for value in enc_patch_sizes)
            dec_depths = tuple(int(value) for value in dec_depths)
            dec_channels = tuple(int(value) for value in dec_channels)
            dec_heads = tuple(int(value) for value in dec_heads)
            dec_patches = tuple(int(value) for value in dec_patch_sizes)
            pool_count = len(enc_depths) - 1
            if pool_count < 1:
                raise ValueError("PTv3 enc_* lists must describe at least two stages")
            if len(dec_depths) != pool_count:
                raise ValueError("PTv3 dec_* lists must have len(enc_*) - 1 entries")
            if strides is None:
                strides = (2,) * pool_count
            else:
                strides = tuple(int(value) for value in strides)
            if len(strides) != pool_count:
                raise ValueError("PTv3 strides must have len(enc_*) - 1 entries")
            if len(enc_depths) != len(enc_channels) or len(enc_depths) != len(enc_heads) or len(enc_depths) != len(enc_patches):
                raise ValueError("PTv3 encoder lists must have identical lengths")
            if len(dec_depths) != len(dec_channels) or len(dec_depths) != len(dec_heads) or len(dec_depths) != len(dec_patches):
                raise ValueError("PTv3 decoder lists must have identical lengths")
            fine_dim = enc_channels[0]
            coarse_dim = enc_channels[-1]
            fine_heads = enc_heads[0]
            coarse_heads = enc_heads[-1]
        else:
            pool_count = max(1, int(round(math.log2(max(1.0, coarse_token_size / fine_token_size)))))
            enc_depths = tuple([fine_depth] + [1] * max(0, pool_count - 1) + [coarse_depth])
            enc_channels = tuple([fine_dim] * pool_count + [coarse_dim])
            enc_heads = tuple([fine_heads] * pool_count + [coarse_heads])
            enc_patches = tuple([fine_window_size] * pool_count + [coarse_window_size])
            dec_depths = tuple([decoder_depth] + [1] * max(0, pool_count - 1))
            dec_channels = tuple([fine_dim] * pool_count)
            dec_heads = tuple([fine_heads] * pool_count)
            dec_patches = tuple([fine_window_size] * pool_count)
            strides = (2,) * pool_count

        if any(value <= 0 for value in (*enc_depths, *enc_channels, *enc_heads, *enc_patches, *dec_depths, *dec_channels, *dec_heads, *dec_patches, *strides)):
            raise ValueError("PTv3 architecture depths, channels, heads, patches, and strides must be positive")
        if any(channels % heads for channels, heads in zip(enc_channels, enc_heads)):
            raise ValueError("every PTv3 encoder channel count must be divisible by its head count")
        if any(channels % heads for channels, heads in zip(dec_channels, dec_heads)):
            raise ValueError("every PTv3 decoder channel count must be divisible by its head count")
        decoder_output_dim = dec_channels[0]

        def build_trunk():
            return PointTransformerV3(
                in_channels=official_input_channels,
                order=serialization_order,
                stride=tuple(strides),
                enc_depths=tuple(enc_depths), enc_channels=tuple(enc_channels),
                enc_num_head=tuple(enc_heads), enc_patch_size=tuple(enc_patches),
                dec_depths=tuple(dec_depths),
                dec_channels=tuple(dec_channels),
                dec_num_head=tuple(dec_heads),
                dec_patch_size=tuple(dec_patches),
                mlp_ratio=mlp_ratio, drop_path=drop_path,
                proj_drop=dropout, attn_drop=dropout,
                qkv_bias=qkv_bias, pre_norm=pre_norm,
                shuffle_orders=shuffle_orders, enable_rpe=enable_rpe,
                enable_flash=enable_flash,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                point_norm=point_norm,
            )

        # The vendored model and serialization code remain official PTv3;
        # only Block construction is wrapped to add DiT-style AdaLN-Zero.
        self.trunk = build_official_ptv3_with_time_context(build_trunk)
        self.head = nn.Sequential(
            nn.LayerNorm(decoder_output_dim),
            nn.Linear(decoder_output_dim, max(1, decoder_output_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, decoder_output_dim // 2), 3),
        )

    @staticmethod
    def _unique_grid(positions, batch_ids, grid_size):
        batch_count = int(batch_ids.max().item()) + 1 if batch_ids.numel() else 1
        minima = positions.new_zeros((batch_count, 3))
        for batch_id in torch.unique(batch_ids, sorted=True):
            mask = batch_ids == batch_id
            minima[batch_id] = positions[mask].amin(dim=0)
        grid = torch.floor((positions - minima[batch_ids]) / grid_size).long()
        keys = torch.cat([batch_ids[:, None], grid], dim=1)
        return torch.unique(keys, dim=0, sorted=True, return_inverse=True)

    def _time_features(self, t, batch_ids, dtype):
        t = t.reshape(-1).to(dtype=dtype)
        frequencies = self.time_frequencies.to(device=t.device, dtype=dtype)
        raw = torch.cat([(t[:, None] * frequencies).sin(), (t[:, None] * frequencies).cos()], dim=-1)
        return raw[batch_ids]

    def _build_features(self, points, voxel_centers, known_flags, extra_features):
        local = points - voxel_centers
        scale = self.coordinate_scale.to(device=points.device, dtype=points.dtype).clamp_min(1e-6)
        features = torch.cat([local, voxel_centers / scale], dim=-1)
        if known_flags is not None:
            features = torch.cat([features, known_flags.to(features.dtype)[:, None]], dim=-1)
        if extra_features is not None:
            features = torch.cat([features, extra_features.to(device=points.device, dtype=points.dtype)], dim=-1)
        if features.shape[-1] != self.input_channels:
            raise ValueError(f"PWM features have {features.shape[-1]} channels, expected {self.input_channels}")
        if self.intra_voxel_encoding_bands:
            frequencies = self.intra_voxel_frequencies.to(device=points.device, dtype=points.dtype)
            scale = self.flow_voxel_size.to(device=points.device, dtype=points.dtype).clamp_min(1e-6)
            angles = (2.0 * local / scale)[..., None] * frequencies
            features = torch.cat([features, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)
        return features

    def forward(self, points, voxel_centers, t, batch_ids=None, extra_features=None):
        return self.forward_conditional(points, voxel_centers, t, batch_ids=batch_ids, extra_features=extra_features)

    def forward_conditional(self, points, voxel_centers, t, batch_ids=None, known_flags=None, extra_features=None):
        if batch_ids is None:
            batch_ids = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        batch_ids = batch_ids.to(device=points.device, dtype=torch.long).reshape(-1)
        if points.ndim != 2 or tuple(points.shape[1:]) != (3,) or voxel_centers.shape != points.shape:
            raise ValueError("points and voxel_centers must both have shape [N, 3]")
        raw_features = self._build_features(points, voxel_centers, known_flags, extra_features)
        fine_keys, point_to_fine = self._unique_grid(points, batch_ids, self.quantization_size)
        fine_count = fine_keys.shape[0]
        fine_points = _segment_mean(points, point_to_fine, fine_count)
        fine_features = _segment_mean(raw_features, point_to_fine, fine_count)
        fine_batch = fine_keys[:, 0]
        token_keys, fine_to_token = self._unique_grid(fine_points, fine_batch, self.fine_token_size)
        token_count = token_keys.shape[0]
        token_points = _segment_mean(fine_points, fine_to_token, token_count)
        token_features = _segment_mean(fine_features, fine_to_token, token_count)
        scene_ids = torch.arange(t.reshape(-1).shape[0], device=points.device)
        time_context = self._time_features(t, scene_ids, points.dtype)
        encoded = self.trunk({
            "coord": token_points, "grid_coord": token_keys[:, 1:].int(),
            "batch": token_keys[:, 0].long(), "feat": token_features,
            "context": time_context,
        })
        return self.head(encoded.feat)[fine_to_token][point_to_fine]


__all__ = ["OfficialPTV3PointFlowNet"]
