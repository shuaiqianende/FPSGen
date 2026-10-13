"""Continuous voxel-density Flow Matching with optional robust supervision.

The primary objective remains ``MSE(predicted velocity, target velocity)``.
Optionally, the reconstructed linear-path endpoint ``y_hat = x_t + (1-t) v``
receives four auxiliary structural terms (see
``docs/3DFRONT_RECHECK_REPORT_2026-08-22.md`` follow-up):

- ``L_fine``: full-resolution L1 in symmetric-log mapped space, restricted to
  a one-voxel dilation of the GT occupancy so thin supports (chair legs) are
  not drowned by empty voxels;
- ``L_s2`` / ``L_s4``: masked-normalized L1 on pooled normalized-count fields
  (``32x64x64 -> 16x32x32`` and ``-> 8x16x16``);
- ``L_grad``: finite-difference gradient matching in mapped space (this is
  matching, not TV regularization).

All defaults keep every weight at zero, preserving historical behavior.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from pointworldmodel.core.conditioning.outpaint import random_known_masks
from pointworldmodel.core.flow_couplings import build_flow_coupling
from pointworldmodel.core.flow_paths import build_flow_path
from pointworldmodel.core.runtime.optimization import build_optimizer_and_scheduler
from pointworldmodel.core.voxel.voxel_unet import VoxelFlowUNet, VoxelLayoutTransformerUNet
from pointworldmodel.core.voxel.grid import VoxelSpec


class DensityFlowModule(LightningModule):
    """Generate continuous voxel density with optional layout/outpainting.

    The default model has one density state channel and one Flow Matching MSE
    objective.  An optional second state channel stores binary voxel support
    (occupied versus empty).  When enabled, support is also transported by
    the same Flow Matching path; its MSE is weighted independently rather than
    being added as a separate geometric loss.

    ``MSE(predicted_velocity, density_target - Gaussian_noise)``.

    Layout conditioning and outpainting only change the model input.  The
    legacy one-channel path remains the default for old configurations and
    checkpoints.
    """

    def __init__(
        self,
        spec=None,
        spec_config=None,
        channels=32,
        lr=2e-4,
        optimization=None,
        target_mapping="symmetric_log",
        count_upper_bound=400.0,
        log_count_scale=None,
        dit_depth=4,
        dit_heads=8,
        mlp_ratio=4.0,
        outpaint=False,
        mask_min_fraction=0.25,
        mask_max_fraction=0.50,
        mask_modes=("directional", "corner"),
        uncond_prob=0.20,
        layout_channels=0,
        condition_type="none",
        condition_count_upper_bound=32.0,
        layout_uncond_prob=0.10,
        layout_uncond_value=0.0,
        layout_encoder_channels=0,
        layout_hi_resolution=None,
        layout_downsample_factor=4,
        architecture="legacy_unet",
        layout_fusion="concat",
        attention_chunk_size=256,
        downsample_levels=3,
        downsample_mode="anisotropic",
        flow_path=None,
        flow_coupling=None,
        lambda_fine=0.0,
        lambda_s2=0.0,
        lambda_s4=0.0,
        lambda_grad=0.0,
        multiscale_layout=False,
        structural_diag_interval_steps=0,
        structural_diag_volume_interval_steps=0,
        soft_dc_enabled=False,
        soft_dc_eta=0.25,
        soft_dc_low_count_max=3.0,
        soft_dc_kappa=1.0,
        mask_channel_enabled=False,
        mask_loss_weight=5.0,
        mask_inference_threshold=0.5,
        mask_apply_inference=True,
    ):
        super().__init__()
        if spec is None:
            spec = VoxelSpec.from_dict(spec_config)
        elif spec_config is not None and spec != VoxelSpec.from_dict(spec_config):
            raise ValueError("spec and spec_config describe different voxel grids")
        if target_mapping not in {"symmetric_log", "legacy_log1p_div_scale"}:
            raise ValueError(f"Unsupported target_mapping: {target_mapping}")
        if target_mapping == "legacy_log1p_div_scale" and log_count_scale is None:
            raise ValueError("log_count_scale is required for legacy mapping")
        if count_upper_bound <= 0:
            raise ValueError("count_upper_bound must be positive")
        if architecture not in {"legacy_unet", "layout_transformer"}:
            raise ValueError("unsupported voxel architecture")
        if downsample_mode not in {"anisotropic", "uniform"}:
            raise ValueError("downsample_mode must be 'anisotropic' or 'uniform'")
        if layout_fusion not in {"concat", "add"}:
            raise ValueError("layout_fusion must be 'concat' or 'add'")
        if layout_fusion == "add" and layout_channels == 0 and layout_encoder_channels == 0:
            raise ValueError("additive layout fusion requires layout channels")
        condition_type = str(condition_type).lower()
        if condition_type == "lidar":
            condition_type = "semantic_kitti_lidar"
        if condition_type not in {"none", "semantic_kitti_lidar"}:
            raise ValueError(
                "condition_type must be 'none' or 'semantic_kitti_lidar'"
            )
        if float(condition_count_upper_bound) <= 0:
            raise ValueError("condition_count_upper_bound must be positive")
        if not 0.0 <= uncond_prob <= 1.0:
            raise ValueError("uncond_prob must lie in [0, 1]")
        if not 0.0 <= layout_uncond_prob <= 1.0:
            raise ValueError("layout_uncond_prob must lie in [0, 1]")
        if not math.isfinite(float(layout_uncond_value)):
            raise ValueError("layout_uncond_value must be finite")
        for name, value in (
            ("lambda_fine", lambda_fine),
            ("lambda_s2", lambda_s2),
            ("lambda_s4", lambda_s4),
            ("lambda_grad", lambda_grad),
        ):
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if any(
            float(value) > 0
            for value in (lambda_fine, lambda_s2, lambda_s4, lambda_grad)
        ) and (flow_path or {}).get("type", "linear") != "linear":
            raise ValueError(
                "structural losses reconstruct the linear-path endpoint and "
                "require flow_path.type: linear"
            )
        if soft_dc_enabled and (flow_path or {}).get("type", "linear") != "linear":
            raise ValueError(
                "soft DC endpoint weighting reconstructs the linear-path endpoint "
                "and requires flow_path.type: linear"
            )
        if not 0.0 < float(soft_dc_eta) <= 1.0:
            raise ValueError("soft_dc_eta must lie in (0, 1]")
        if float(soft_dc_low_count_max) < 0.0:
            raise ValueError("soft_dc_low_count_max must be non-negative")
        if float(soft_dc_kappa) < 0.0:
            raise ValueError("soft_dc_kappa must be non-negative")
        if not 0.0 < float(mask_loss_weight):
            raise ValueError("mask_loss_weight must be positive")
        if not 0.0 <= float(mask_inference_threshold) <= 1.0:
            raise ValueError("mask_inference_threshold must lie in [0, 1]")
        if mask_channel_enabled and soft_dc_enabled:
            raise ValueError(
                "mask_channel_enabled is currently only supported with pure FM MSE; "
                "disable soft_dc for the mask-channel experiment"
            )
        if mask_channel_enabled and any(
            float(value) > 0
            for value in (lambda_fine, lambda_s2, lambda_s4, lambda_grad)
        ):
            raise ValueError(
                "mask_channel_enabled is currently only supported with pure FM MSE; "
                "disable structural auxiliary losses for the mask-channel experiment"
            )
        if layout_encoder_channels > 0:
            layout_channels = layout_encoder_channels

        spec_config = spec.to_dict()
        self.save_hyperparameters(ignore=["spec"])
        self.spec = spec
        self.flow_path = build_flow_path(
            flow_path,
            target_mapping=target_mapping,
            count_upper_bound=count_upper_bound,
            log_count_scale=log_count_scale,
        )
        self.flow_coupling = build_flow_coupling(flow_coupling)
        self._last_coupling_diagnostics = {}
        self._last_soft_dc_diagnostics = {}
        self._last_channel_diagnostics = {}

        if layout_encoder_channels > 0:
            from pointworldmodel.core.conditioning.layout_encoder import LayoutEncoder

            self.layout_encoder = LayoutEncoder(
                in_channels=5,
                out_channels=layout_encoder_channels,
                hi_resolution=layout_hi_resolution or (512, 512, 48),
                output_stride=layout_downsample_factor,
            )
        else:
            self.layout_encoder = None

        # ``field_channels`` is the number of transported fields.  Outpainting
        # additionally supplies a known target with the same number of fields
        # and one binary known-region indicator.
        field_channels = 2 if mask_channel_enabled else 1
        network_input_channels = field_channels + (
            field_channels + 1 if outpaint else 0
        )
        self.field_channels = field_channels

        if architecture == "legacy_unet":
            self.field_feature_projection = (
                nn.Conv3d(network_input_channels, layout_channels, kernel_size=1)
                if layout_fusion == "add"
                else None
            )
            self.net = VoxelFlowUNet(
                in_channels=layout_channels
                if layout_fusion == "add"
                else network_input_channels + layout_channels,
                out_channels=field_channels,
                base_channels=channels,
                dit_depth=dit_depth,
                dit_heads=dit_heads,
                mlp_ratio=mlp_ratio,
                downsample_levels=downsample_levels,
                downsample_mode=downsample_mode,
            )
        else:
            self.field_feature_projection = None
            self.net = VoxelLayoutTransformerUNet(
                in_channels=network_input_channels,
                out_channels=field_channels,
                layout_channels=layout_channels,
                base_channels=channels,
                dit_depth=dit_depth,
                dit_heads=dit_heads,
                mlp_ratio=mlp_ratio,
                attention_chunk_size=attention_chunk_size,
                multiscale_layout_injection=multiscale_layout,
            )

    def on_train_epoch_start(self):
        """Reset stateful coupling buffers before each shuffled epoch."""

        reset = getattr(self.flow_coupling, "reset", None)
        if reset is not None:
            reset()

    def encode_sparse_layout(self, layout_coords, layout_feats, batch_size):
        if self.layout_encoder is None:
            raise ValueError("model has no sparse layout encoder")
        if (layout_coords is None) != (layout_feats is None):
            raise ValueError("layout_coords and layout_feats must be provided together")
        if layout_coords is None:
            return None
        if torch.is_tensor(layout_coords):
            layout_coords = [layout_coords]
        if torch.is_tensor(layout_feats):
            layout_feats = [layout_feats]
        if len(layout_coords) != batch_size or len(layout_feats) != batch_size:
            raise ValueError("sparse layout batch size does not match voxel batch")
        return self.layout_encoder(layout_coords, layout_feats, batch_size=batch_size)

    def prepare_layout_condition(
        self,
        layout,
        batch_size,
        dtype,
        *,
        layout_coords=None,
        layout_feats=None,
        dropout=False,
    ):
        if layout_coords is not None or layout_feats is not None:
            if layout is not None:
                raise ValueError("provide dense or sparse layout, not both")
            layout = self.encode_sparse_layout(layout_coords, layout_feats, batch_size)
        if self.hparams.layout_channels == 0:
            if layout is not None:
                raise ValueError("model was created without layout channels")
            return None
        expected = (
            batch_size,
            self.hparams.layout_channels,
            self.spec.nz,
            self.spec.ny,
            self.spec.nx,
        )
        if layout is None:
            layout = torch.full(
                expected,
                float(self.hparams.layout_uncond_value),
                device=self.device,
                dtype=dtype,
            )
        else:
            layout = torch.as_tensor(layout, device=self.device, dtype=dtype)
            if tuple(layout.shape) != expected:
                raise ValueError(f"layout has shape {tuple(layout.shape)}, expected {expected}")
        if dropout and self.hparams.layout_uncond_prob > 0:
            dropped = torch.rand(batch_size, device=layout.device) < self.hparams.layout_uncond_prob
            layout = layout.clone()
            layout[dropped] = float(self.hparams.layout_uncond_value)
        return layout

    def _predict_velocity(self, state, t, layout):
        if self.hparams.architecture == "layout_transformer":
            return self.net(state, t, layout=layout)
        if self.hparams.layout_fusion == "add":
            if layout is None:
                raise ValueError("additive fusion requires layout")
            state = self.field_feature_projection(state) + layout
        elif layout is not None:
            state = torch.cat([state, layout], dim=1)
        return self.net(state, t)

    def counts_to_target(self, counts):
        counts = counts.float().clamp_min(0)
        if self.hparams.target_mapping == "symmetric_log":
            scale = counts.new_tensor(math.log1p(self.hparams.count_upper_bound))
            return 2.0 * torch.log1p(counts.clamp_max(self.hparams.count_upper_bound)) / scale - 1.0
        return torch.log1p(counts) / self.hparams.log_count_scale

    def target_to_density(self, target):
        if self.hparams.target_mapping == "symmetric_log":
            scale = target.new_tensor(math.log1p(self.hparams.count_upper_bound))
            return torch.expm1(0.5 * (target.clamp(-1.0, 1.0) + 1.0) * scale)
        return torch.expm1((target * self.hparams.log_count_scale).clamp_min(0))

    def count_field(self, points):
        return self.counts_to_target(self.spec.counts(points)).view(
            self.spec.nz, self.spec.ny, self.spec.nx
        )

    def raw_count_field(self, points):
        """Return the un-mapped per-voxel count field for path context."""
        return self.spec.counts(points).float().view(
            self.spec.nz, self.spec.ny, self.spec.nx
        )

    def _build_flow_target(self, counts):
        """Build the transported density (and optional support) fields."""
        density_target = self.counts_to_target(counts)
        if not self.hparams.mask_channel_enabled:
            return density_target
        support_target = (counts > 0).to(density_target.dtype)
        return torch.cat([density_target, support_target], dim=1)

    def _flow_mse_by_channel(self, residual, active_mask=None):
        """Return pure FM MSE with an optional support-channel multiplier.

        Each channel is normalized by its own active voxel count, so
        ``mask_loss_weight=5`` means ``density_mse + 5 * mask_mse`` rather
        than an accidental weight diluted by the number of channels.
        """
        if active_mask is None:
            active = torch.ones(
                (residual.shape[0], 1, *residual.shape[2:]),
                device=residual.device,
                dtype=residual.dtype,
            )
        else:
            active = active_mask.to(residual.dtype)
        channel_mass = active.sum(dim=(0, 2, 3, 4)).clamp_min(1.0)
        channel_mse = (residual * active).sum(dim=(0, 2, 3, 4)) / channel_mass
        if residual.shape[1] == 1:
            weights = residual.new_ones(1)
        else:
            weights = residual.new_ones(residual.shape[1])
            weights[1] = float(self.hparams.mask_loss_weight)
        loss = (channel_mse * weights).sum()
        self._last_channel_diagnostics = {
            "density_mse": channel_mse[0].detach(),
            "mask_mse": (
                channel_mse[1].detach()
                if residual.shape[1] > 1
                else residual.new_zeros(())
            ),
        }
        return loss

    def flow_loss(
        self,
        point_clouds,
        layout=None,
        *,
        condition=None,
        layout_coords=None,
        layout_feats=None,
    ):
        self._last_channel_diagnostics = {}
        if condition is not None:
            if layout is not None or layout_coords is not None or layout_feats is not None:
                raise ValueError("provide condition or layout, not both")
            layout = condition
        counts = torch.stack(
            [self.raw_count_field(points) for points in point_clouds], dim=0
        )[:, None]
        target = self._build_flow_target(counts)
        layout = self.prepare_layout_condition(
            layout,
            target.shape[0],
            target.dtype,
            layout_coords=layout_coords,
            layout_feats=layout_feats,
            dropout=self.training,
        )
        noise = torch.randn_like(target)
        coupling_output = self.flow_coupling.pair(
            x0=noise,
            x1=target,
            context={"voxel_counts": counts},
        )
        self._last_coupling_diagnostics = coupling_output.diagnostics
        t = torch.rand((target.shape[0],), device=target.device)
        path_output = self.flow_path.sample(
            x0=coupling_output.x0,
            x1=coupling_output.x1,
            t=t,
            context={"voxel_counts": counts},
        )
        state = path_output.xt
        velocity = path_output.velocity
        noisy_state = state
        unknown = None

        if not self.hparams.outpaint:
            prediction = self._predict_velocity(state, t, layout)
            residual = (prediction - velocity).square()
            if self.hparams.mask_channel_enabled:
                loss = self._flow_mse_by_channel(residual)
                soft_dc_weights = residual.new_ones(residual.shape)
            else:
                loss, soft_dc_weights = self._weighted_flow_mse(
                    residual, noisy_state, prediction, t, counts, None
                )
        else:
            known_mask = random_known_masks(
                target.shape[0],
                self.spec,
                target.device,
                self.hparams.mask_min_fraction,
                self.hparams.mask_max_fraction,
                self.hparams.mask_modes,
            )
            unconditional = torch.rand(target.shape[0], device=target.device) < self.hparams.uncond_prob
            known_mask[unconditional] = False
            known_target = target * known_mask
            state = torch.where(known_mask, target, state)
            network_input = torch.cat([state, known_target, known_mask.to(target.dtype)], dim=1)
            prediction = self._predict_velocity(network_input, t, layout)
            unknown = (~known_mask).to(target.dtype)
            residual = (prediction - velocity).square()
            if self.hparams.mask_channel_enabled:
                loss = self._flow_mse_by_channel(residual, unknown)
                soft_dc_weights = unknown.expand_as(residual)
            else:
                loss, soft_dc_weights = self._weighted_flow_mse(
                    residual, noisy_state, prediction, t, counts, unknown
                )

        if self.hparams.soft_dc_enabled:
            relaxed = (soft_dc_weights < 1.0 - 1.0e-6).to(soft_dc_weights.dtype)
            active = soft_dc_weights > 0
            self._last_soft_dc_diagnostics = {
                "inside_fraction": relaxed[active].mean() if active.any() else relaxed.new_zeros(()),
                "active_fraction": active.to(soft_dc_weights.dtype).mean(),
                "weight_mean": soft_dc_weights[active].mean() if active.any() else soft_dc_weights.new_zeros(()),
                "base_mse": residual.mul(unknown if unknown is not None else 1.0).sum()
                / (unknown.sum() if unknown is not None else residual.numel()),
            }
        else:
            self._last_soft_dc_diagnostics = {}

        auxiliary = self._structural_losses(
            noisy_state, prediction, t, target, counts, unknown
        )
        weights = {
            "fine": self.hparams.lambda_fine,
            "s2": self.hparams.lambda_s2,
            "s4": self.hparams.lambda_s4,
            "grad": self.hparams.lambda_grad,
        }
        for name, value in auxiliary.items():
            self.log(f"train/structural_{name}", value.detach(), on_step=True, on_epoch=False)
            loss = loss + float(weights[name]) * value
        return loss

    def _weighted_flow_mse(self, residual, noisy_state, prediction, t, counts, active_mask):
        """Compute FM MSE with optional raw-count interval robustness.

        The GT count field is never changed. For the linear path, the
        network's implied endpoint is ``x_t + (1-t) * v_pred``. For counts
        above ``soft_dc_low_count_max``, an endpoint inside a local
        ``C +/- kappa*sqrt(C)`` count interval receives only ``eta`` weight.
        Low-count voxels retain full weight so thin structures and occupancy
        are not relaxed. Interval membership is detached from autograd; the
        FM residual remains fully differentiable.
        """
        if not self.hparams.soft_dc_enabled:
            weights = (
                residual.new_ones(residual.shape)
                if active_mask is None
                else active_mask.to(residual.dtype)
            )
            return residual.mul(weights).sum() / weights.sum().clamp_min(1.0), weights

        t_view = t.view(-1, 1, 1, 1, 1).to(noisy_state.dtype)
        endpoint = noisy_state + (1.0 - t_view) * prediction
        predicted_counts = self.target_to_density(endpoint.clamp(-1.0, 1.0))

        low_count = counts <= float(self.hparams.soft_dc_low_count_max)
        safe_counts = counts.clamp_min(1.0)
        half_width = float(self.hparams.soft_dc_kappa) * safe_counts.sqrt()
        lower = (counts - half_width).clamp_min(0.0)
        upper = counts + half_width
        inside = (predicted_counts >= lower) & (predicted_counts <= upper)
        relaxed = inside & ~low_count
        weights = torch.where(
            relaxed,
            torch.full_like(predicted_counts, float(self.hparams.soft_dc_eta)),
            torch.ones_like(predicted_counts),
        )
        if active_mask is not None:
            weights = weights * active_mask.to(weights.dtype)
        weights = weights.detach()
        return residual.mul(weights).sum() / weights.sum().clamp_min(1.0), weights

    def _structural_losses(self, noisy_state, prediction, t, target, counts, unknown):
        """Auxiliary Fine/S2/S4/Gradient losses on the reconstructed endpoint.

        ``y_hat = x_t + (1 - t) v_pred`` is exact for the linear path.  Fine
        and gradient terms stay in symmetric-log mapped space (sensitive to
        the empty <-> one-point distinction); S2/S4 convert to normalized
        count space (regional mass).  Under outpainting every term is masked
        to the unknown region, which equals the full grid for unconditional
        samples.
        """
        weights = (
            self.hparams.lambda_fine,
            self.hparams.lambda_s2,
            self.hparams.lambda_s4,
            self.hparams.lambda_grad,
        )
        if not any(float(w) > 0 for w in weights):
            return {}
        t_view = t.view(-1, 1, 1, 1, 1).to(noisy_state.dtype)
        y_hat = noisy_state + (1.0 - t_view) * prediction
        losses = {}
        geometry_roi = None
        if self.hparams.lambda_fine > 0 or self.hparams.lambda_grad > 0:
            gt_occ = (counts > 0).to(y_hat.dtype)
            geometry_roi = F.max_pool3d(gt_occ, kernel_size=3, stride=1, padding=1)
        fine_mask = None
        if self.hparams.lambda_fine > 0 or self.hparams.lambda_grad > 0:
            fine_mask = geometry_roi if unknown is None else geometry_roi * unknown
        if self.hparams.lambda_fine > 0:
            losses["fine"] = (
                fine_mask * (y_hat - target).abs()
            ).sum() / fine_mask.sum().clamp_min(1.0)
        if self.hparams.lambda_s2 > 0 or self.hparams.lambda_s4 > 0:
            upper = self.hparams.count_upper_bound
            pred_norm = self.target_to_density(y_hat.clamp(-1.0, 1.0)) / upper
            gt_norm = counts.clamp(min=0.0, max=upper) / upper
            mask = unknown if unknown is not None else torch.ones_like(y_hat)
            if self.hparams.lambda_s2 > 0:
                losses["s2"] = self._masked_pool_l1(pred_norm, gt_norm, mask, 2)
            if self.hparams.lambda_s4 > 0:
                losses["s4"] = self._masked_pool_l1(pred_norm, gt_norm, mask, 4)
        if self.hparams.lambda_grad > 0:
            losses["grad"] = self._gradient_matching_l1(y_hat, target, fine_mask)
        return losses

    @staticmethod
    def _masked_pool_l1(pred, gt, mask, factor):
        """Masked-normalized pooled L1.

        Dividing by the pooled mask keeps blocks that mix known and unknown
        voxels from being diluted by their known fraction.
        """
        eps = 1e-6
        mask_s = F.avg_pool3d(mask, factor, stride=factor)
        pred_s = F.avg_pool3d(pred * mask, factor, stride=factor) / mask_s.clamp_min(eps)
        gt_s = F.avg_pool3d(gt * mask, factor, stride=factor) / mask_s.clamp_min(eps)
        return (mask_s * (pred_s - gt_s).abs()).sum() / mask_s.sum().clamp_min(eps)

    @staticmethod
    def _gradient_matching_l1(y_hat, target, mask):
        """Masked finite-difference gradient matching (not TV smoothing)."""
        total = y_hat.new_zeros(())
        for dim in (-3, -2, -1):
            d_pred = y_hat.diff(dim=dim)
            d_gt = target.diff(dim=dim)
            pair_mask = mask.narrow(dim, 1, mask.shape[dim] - 1) * mask.narrow(
                dim, 0, mask.shape[dim] - 1
            )
            total = total + (pair_mask * (d_pred - d_gt).abs()).sum() / pair_mask.sum().clamp_min(1.0)
        return total

    def training_step(self, batch, _):
        points = [item.to(self.device) for item in batch["points"]]
        condition = batch.get("condition")
        if condition is None and self.hparams.condition_type == "semantic_kitti_lidar":
            from pointworldmodel.core.data.semantic_kitti import (
                semantic_kitti_condition_from_batch,
            )

            condition = semantic_kitti_condition_from_batch(
                batch,
                self.spec,
                device=self.device,
                dtype=points[0].dtype,
                count_upper_bound=float(self.hparams.condition_count_upper_bound),
            )
        loss = self.flow_loss(
            points,
            batch.get("layout"),
            condition=condition,
            layout_coords=batch.get("layout_coords"),
            layout_feats=batch.get("layout_feats"),
        )
        self.log("train/voxel_density_mse", loss, prog_bar=True)
        for name, value in self._last_soft_dc_diagnostics.items():
            self.log(
                f"train/soft_dc_{name}",
                value.detach(),
                on_step=True,
                on_epoch=False,
            )
        for name, value in self._last_channel_diagnostics.items():
            self.log(
                f"train/mask_channel_{name}",
                value,
                on_step=True,
                on_epoch=False,
            )
        for name, value in self._last_coupling_diagnostics.items():
            if name == "source_for_target" or not torch.is_tensor(value) or value.ndim != 0:
                continue
            self.log(
                f"flow_coupling/{name}",
                value.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        diag_interval = self.hparams.structural_diag_interval_steps
        if diag_interval > 0 and self.global_step > 0 and self.global_step % diag_interval == 0:
            volume_interval = self.hparams.structural_diag_volume_interval_steps
            save_volume = (
                volume_interval > 0 and self.global_step % volume_interval == 0
            )
            self._thin_support_diagnostics(batch, save_volume=save_volume)
        return loss

    @torch.no_grad()
    def _thin_support_diagnostics(self, batch, *, save_volume=False, steps=50):
        """Distinguish missing support from allocation-side removal.

        Samples one scene unconditionally (layout-only), allocates the GT
        point total over the predicted density, and compares against GT
        occupancy.  Situation A (predicted near-empty at GT leg voxels) means
        the network lacks support; situation B (weak density but zero
        allocated count) means the allocation/threshold post-processing
        removes thin structure the network did produce.
        """
        from pointworldmodel.core.voxel.grid import allocate_counts

        gt_counts = self.raw_count_field(batch["points"][0].to(self.device))
        gt_flat = gt_counts.reshape(-1)
        total_points = int(gt_flat.sum().item())
        layout = batch.get("layout")
        layout_item = None
        if layout is not None:
            layout_item = layout[0:1] if torch.is_tensor(layout) else [layout[0]]
        density = self.sample_density(
            steps=steps,
            batch_size=1,
            solver="euler",
            layout=layout_item,
            layout_coords=(
                [batch["layout_coords"][0]] if batch.get("layout_coords") is not None else None
            ),
            layout_feats=(
                [batch["layout_feats"][0]] if batch.get("layout_feats") is not None else None
            ),
        )[0]
        allocated = allocate_counts(density, total_points)

        gt_occ = gt_flat > 0
        near_empty = density < 0.1
        gt_occ_count = int(gt_occ.sum().item())
        thin = gt_occ & (gt_flat <= 8)
        stats = {
            "diag/gt_occupied_count": float(gt_occ_count),
            "diag/gt_occupied_pred_near_empty_ratio": (
                float((gt_occ & near_empty).sum().item()) / max(gt_occ_count, 1)
            ),
            "diag/gt_occupied_allocated_zero_ratio": (
                float((gt_occ & (allocated == 0)).sum().item()) / max(gt_occ_count, 1)
            ),
            "diag/thin_support_voxel_count": float(int(thin.sum().item())),
            "diag/thin_support_recall": (
                float((thin & (allocated > 0)).sum().item()) / max(int(thin.sum().item()), 1)
            ),
        }
        for name, value in stats.items():
            self.log(name, value, on_step=True, on_epoch=False)

        if save_volume and self.trainer is not None:
            out_dir = Path(self.trainer.default_root_dir) / "structural_diag"
            out_dir.mkdir(parents=True, exist_ok=True)
            grid = (self.spec.nz, self.spec.ny, self.spec.nx)
            np.savez_compressed(
                out_dir / f"step_{self.global_step:07d}.npz",
                gt_counts=gt_flat.cpu().numpy().reshape(grid).astype(np.int32),
                predicted_density=density.cpu().numpy().reshape(grid).astype(np.float32),
                allocated_counts=allocated.cpu().numpy().reshape(grid).astype(np.int32),
                stats=np.array([stats["diag/gt_occupied_pred_near_empty_ratio"]]),
            )

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            self,
            lr=self.hparams.lr,
            optimization=self.hparams.optimization,
        )

    @staticmethod
    def _as_grid(value, batch_size, spec, device, dtype):
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 1:
            if batch_size != 1:
                raise ValueError("a flattened grid can only describe batch_size=1")
            value = value.view(1, 1, spec.nz, spec.ny, spec.nx)
        elif value.ndim == 2:
            value = value.view(batch_size, 1, spec.nz, spec.ny, spec.nx)
        elif value.ndim == 3:
            if batch_size != 1:
                raise ValueError("a 3-D grid can only describe batch_size=1")
            value = value.view(1, 1, spec.nz, spec.ny, spec.nx)
        elif value.ndim == 4:
            value = value[:, None]
        elif value.ndim != 5:
            raise ValueError("grid must have 1, 2, 3, 4, or 5 dimensions")
        expected = (batch_size, 1, spec.nz, spec.ny, spec.nx)
        if tuple(value.shape) != expected:
            raise ValueError(f"grid has shape {tuple(value.shape)}, expected {expected}")
        return value

    @torch.no_grad()
    def _sample_state(
        self,
        steps=32,
        batch_size=1,
        solver="euler",
        known_counts=None,
        known_mask=None,
        layout=None,
        condition=None,
        layout_coords=None,
        layout_feats=None,
    ):
        if steps <= 0 or solver not in {"euler", "heun"}:
            raise ValueError("steps must be positive and solver must be euler or heun")
        state = torch.randn(
            (batch_size, self.field_channels, self.spec.nz, self.spec.ny, self.spec.nx),
            device=self.device,
        )
        if condition is not None:
            if layout is not None or layout_coords is not None or layout_feats is not None:
                raise ValueError("provide condition or layout, not both")
            layout = condition
        layout = self.prepare_layout_condition(
            layout,
            batch_size,
            state.dtype,
            layout_coords=layout_coords,
            layout_feats=layout_feats,
        )
        if self.hparams.outpaint:
            if known_counts is None:
                known_counts = torch.zeros(
                    (batch_size, 1, self.spec.nz, self.spec.ny, self.spec.nx),
                    device=state.device,
                    dtype=state.dtype,
                )
            else:
                known_counts = self._as_grid(
                    known_counts, batch_size, self.spec, state.device, state.dtype
                )
            if known_mask is None:
                known_mask = torch.zeros_like(known_counts, dtype=torch.bool)
            else:
                known_mask = self._as_grid(
                    known_mask, batch_size, self.spec, state.device, torch.bool
                )
            known_density = self.counts_to_target(known_counts) * known_mask
            if self.hparams.mask_channel_enabled:
                known_support = (known_counts > 0).to(state.dtype) * known_mask
                known_target = torch.cat([known_density, known_support], dim=1)
            else:
                known_target = known_density
            state = torch.where(known_mask, known_target, state)
        elif known_counts is not None or known_mask is not None:
            raise ValueError("checkpoint was not trained for outpaint")

        dt = 1.0 / steps
        for index in range(steps):
            t = torch.full((batch_size,), index / steps, device=state.device)
            if self.hparams.outpaint:
                inp = torch.cat([state, known_target, known_mask.to(state.dtype)], dim=1)
                velocity = self._predict_velocity(inp, t, layout) * (~known_mask)
            else:
                velocity = self._predict_velocity(state, t, layout)
            if solver == "heun":
                predicted = state + dt * velocity
                if self.hparams.outpaint:
                    predicted = torch.where(known_mask, known_target, predicted)
                    next_input = torch.cat(
                        [predicted, known_target, known_mask.to(state.dtype)], dim=1
                    )
                    t_next = torch.full(
                        (batch_size,), (index + 1) / steps, device=state.device
                    )
                    velocity_next = self._predict_velocity(
                        next_input, t_next, layout
                    ) * (~known_mask)
                else:
                    t_next = torch.full(
                        (batch_size,), (index + 1) / steps, device=state.device
                    )
                    velocity_next = self._predict_velocity(predicted, t_next, layout)
                state = state + 0.5 * dt * (velocity + velocity_next)
            else:
                state = state + dt * velocity
            if self.hparams.outpaint:
                state = torch.where(known_mask, known_target, state)
        return state

    @torch.no_grad()
    def sample_density_and_mask(self, **kwargs):
        """Sample density and, for mask-channel models, the support probability.

        The mask is decoded from the second transported state channel and
        clamped to the endpoint range.  It is returned without the hard
        density threshold used by evaluation scripts.
        """
        state = self._sample_state(**kwargs)
        density = self.target_to_density(state[:, 0])
        if not self.hparams.mask_channel_enabled:
            return density.flatten(1), None
        support = state[:, 1].clamp(0.0, 1.0)
        if self.hparams.mask_apply_inference:
            density = torch.where(
                support >= float(self.hparams.mask_inference_threshold),
                density,
                torch.zeros_like(density),
            )
        return density.flatten(1), support.flatten(1)

    @torch.no_grad()
    def sample_density(self, **kwargs):
        """Backward-compatible density-only sampling API."""
        density, _ = self.sample_density_and_mask(**kwargs)
        return density
