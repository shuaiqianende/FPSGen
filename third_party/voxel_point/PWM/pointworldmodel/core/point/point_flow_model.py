"""PointFlow student for the unified voxel-anchor transport pipeline.

Training has two explicit stages:

1. :class:`PointMappingTeacher` learns a source-indexed endpoint from the
   voxel-centre Gaussian anchors to the GT points using only Chamfer distance.
2. This module learns the flow field from the same anchors to the frozen
   teacher endpoints with ordinary flow-matching MSE.

The student never receives GT points in its network input.  Layout and
outpainting are optional dataset conditions and do not change the transport
objective.
"""

from __future__ import annotations

import gc
import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from pointworldmodel.core.point.anchor import sample_gaussian_anchors
from pointworldmodel.core.voxel.grid import VoxelSpec
from pointworldmodel.core.conditioning.outpaint import point_known_flags, random_known_masks
from pointworldmodel.core.point.ptv3_active_network import OfficialPTV3PointFlowNet
from pointworldmodel.core.runtime.optimization import build_optimizer_and_scheduler


class PointFlowModule(LightningModule):
    """Distil a frozen CD teacher into a conditional point flow field."""

    def __init__(
        self,
        spec=None,
        spec_config=None,
        lr=2e-4,
        optimization=None,
        teacher_checkpoint=None,
        anchor_noise_std_ratio=0.25,
        anchor_clip_to_voxel=False,
        endpoint_refiner=None,
        clear_teacher_cache=True,
        base_channels=32,
        quantization_size=0.1,
        backbone="ptv3",
        fine_token_size=0.10,
        coarse_token_size=0.20,
        intra_voxel_encoding_bands=4,
        fine_dim=None,
        coarse_dim=None,
        fine_depth=3,
        coarse_depth=4,
        decoder_depth=2,
        fine_heads=8,
        coarse_heads=8,
        fine_window_size=64,
        coarse_window_size=128,
        dropout=0.0,
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
        point_norm="batch",
        serialization_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders=True,
        mlp_ratio=4.0,
        drop_path=0.3,
        pre_norm=True,
        qkv_bias=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_depths=None,
        enc_channels=None,
        enc_heads=None,
        enc_patch_sizes=None,
        dec_depths=None,
        dec_channels=None,
        dec_heads=None,
        dec_patch_sizes=None,
        strides=None,
    ):
        super().__init__()
        if spec is None:
            spec = VoxelSpec.from_dict(spec_config)
        elif spec_config is not None and spec != VoxelSpec.from_dict(spec_config):
            raise ValueError("spec and spec_config describe different voxel grids")
        if not teacher_checkpoint:
            raise ValueError("teacher_checkpoint is required for PointFlow training")
        if backbone not in {"ptv3", "ptv3_official"}:
            raise ValueError(
                "the unified PointFlow implementation uses the official "
                "PTv3 backbone ('ptv3_official')"
            )
        if anchor_noise_std_ratio < 0:
            raise ValueError("anchor_noise_std_ratio must be non-negative")
        if not 0.0 <= uncond_prob <= 1.0:
            raise ValueError("uncond_prob must lie in [0, 1]")
        if layout_channels < 0 or layout_encoder_channels < 0:
            raise ValueError("layout channel counts must be non-negative")
        condition_type = str(condition_type).lower()
        if condition_type == "lidar":
            condition_type = "semantic_kitti_lidar"
        if condition_type not in {"none", "semantic_kitti_lidar"}:
            raise ValueError(
                "condition_type must be 'none' or 'semantic_kitti_lidar'"
            )
        if float(condition_count_upper_bound) <= 0:
            raise ValueError("condition_count_upper_bound must be positive")
        if not 0.0 <= layout_uncond_prob <= 1.0:
            raise ValueError("layout_uncond_prob must lie in [0, 1]")
        if not math.isfinite(float(layout_uncond_value)):
            raise ValueError("layout_uncond_value must be finite")
        if layout_encoder_channels > 0:
            layout_channels = layout_encoder_channels

        spec_config = spec.to_dict()
        self.save_hyperparameters(ignore=["spec"])
        self.spec = spec

        from pointworldmodel.core.point.endpoint_refinement import build_endpoint_refiner

        # Old checkpoints/configs have no endpoint_refiner field and therefore
        # retain the exact Teacher -> Student identity-target behavior.
        self.endpoint_refiner = build_endpoint_refiner(endpoint_refiner)

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

        coordinate_scale = (
            (spec.x_max - spec.x_min) / 2.0,
            (spec.y_max - spec.y_min) / 2.0,
            (spec.z_max - spec.z_min) / 2.0,
        )
        # The active backbone is the vendored official PTv3 core.  The old
        # ``point_flow_network.PTV3PointFlowNet`` remains importable only for
        # historical code and must not be used for new checkpoints.
        self.net = OfficialPTV3PointFlowNet(
            in_channels=(7 if outpaint else 6) + layout_channels,
            base_channels=base_channels,
            quantization_size=quantization_size,
            coordinate_scale=coordinate_scale,
            fine_token_size=fine_token_size,
            coarse_token_size=coarse_token_size,
            flow_voxel_size=spec.size,
            intra_voxel_encoding_bands=intra_voxel_encoding_bands,
            fine_dim=fine_dim,
            coarse_dim=coarse_dim,
            fine_depth=fine_depth,
            coarse_depth=coarse_depth,
            decoder_depth=decoder_depth,
            fine_heads=fine_heads,
            coarse_heads=coarse_heads,
            fine_window_size=fine_window_size,
            coarse_window_size=coarse_window_size,
            dropout=dropout,
            point_norm=point_norm,
            serialization_order=serialization_order,
            shuffle_orders=shuffle_orders,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            pre_norm=pre_norm,
            qkv_bias=qkv_bias,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_heads=enc_heads,
            enc_patch_sizes=enc_patch_sizes,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_heads=dec_heads,
            dec_patch_sizes=dec_patch_sizes,
            strides=strides,
        )

        from pointworldmodel.core.point.mapping_teacher import PointMappingTeacher

        self.mapping_teacher = PointMappingTeacher.load_from_checkpoint(
            teacher_checkpoint,
            map_location="cpu",
            clear_cuda_cache=clear_teacher_cache,
        )
        if self.mapping_teacher.spec != self.spec:
            raise ValueError("teacher and student checkpoints use different VoxelSpec values")
        if bool(self.mapping_teacher.hparams.outpaint) != bool(outpaint):
            raise ValueError("teacher and student must agree on model.outpaint")
        teacher_ratio = float(
            getattr(self.mapping_teacher.hparams, "anchor_noise_std_ratio", 0.25)
        )
        if abs(teacher_ratio - float(anchor_noise_std_ratio)) > 1e-8:
            raise ValueError(
                "teacher and student must use the same anchor_noise_std_ratio"
            )
        teacher_clip = bool(
            getattr(self.mapping_teacher.hparams, "anchor_clip_to_voxel", False)
        )
        if teacher_clip != bool(anchor_clip_to_voxel):
            raise ValueError("teacher and student must use the same anchor clipping policy")
        self.mapping_teacher.eval()
        for parameter in self.mapping_teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        """Keep the frozen teacher in evaluation mode during student training."""
        super().train(mode)
        self.mapping_teacher.eval()
        return self

    def encode_sparse_layout(self, layout_coords, layout_feats, batch_size):
        """Encode high-resolution sparse layout into the model lattice."""
        if self.layout_encoder is None:
            raise ValueError("this model was created without a sparse layout encoder")
        if (layout_coords is None) != (layout_feats is None):
            raise ValueError("layout_coords and layout_feats must be provided together")
        if layout_coords is None:
            return None
        if torch.is_tensor(layout_coords):
            layout_coords = [layout_coords]
        if torch.is_tensor(layout_feats):
            layout_feats = [layout_feats]
        if len(layout_coords) != batch_size or len(layout_feats) != batch_size:
            raise ValueError("sparse layout batch size does not match point batch")
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
        """Normalize dense and sparse layout inputs for train and sampling."""
        if layout_coords is not None or layout_feats is not None:
            if layout is not None:
                raise ValueError("provide either dense layout or sparse layout, not both")
            layout = self.encode_sparse_layout(layout_coords, layout_feats, batch_size)
        if self.hparams.layout_channels == 0:
            if layout is not None:
                raise ValueError("this model was created without layout channels")
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

    def _layout_features(self, layout, batch_ids, voxel_ids):
        if layout is None:
            return None
        features = layout.permute(0, 2, 3, 4, 1).reshape(
            layout.shape[0], self.spec.num_voxels, self.hparams.layout_channels
        )
        return features[batch_ids.long(), voxel_ids.long()]

    def _sample_valid_known_flags(self, target_ids, max_attempts=64):
        """Sample an outpaint crop containing both known and unknown points."""
        minimum_points = min(1024, max(1, target_ids.numel() // 100))
        for _ in range(max_attempts):
            known_mask = random_known_masks(
                1,
                self.spec,
                target_ids.device,
                self.hparams.mask_min_fraction,
                self.hparams.mask_max_fraction,
                self.hparams.mask_modes,
            )
            flags = point_known_flags(target_ids, known_mask)
            if (
                int(flags.sum().item()) >= minimum_points
                and int((~flags).sum().item()) >= minimum_points
            ):
                return flags

        # Deterministic fallback for a compact object occupying only a few
        # BEV cells.  This keeps the outpaint loss defined without changing
        # the sampled geometry.
        for axis in (
            target_ids % self.spec.nx,
            (target_ids // self.spec.nx) % self.spec.ny,
        ):
            values, counts = torch.unique(axis, sorted=True, return_counts=True)
            if values.numel() < 2:
                continue
            cumulative = counts.cumsum(0)[:-1]
            split = torch.argmin((cumulative - target_ids.numel() / 2).abs())
            flags = axis <= values[split]
            if flags.any() and (~flags).any():
                return flags
        return torch.zeros_like(target_ids, dtype=torch.bool)

    def _scene_pair(self, target):
        """Prepare unknown targets, teacher context, and Gaussian anchors."""
        target_ids, valid = self.spec.point_to_indices(target)
        target = target[valid]
        target_ids = target_ids[valid]
        if target.shape[0] == 0:
            raise ValueError("a point cloud contains no points inside the voxel grid")

        context = target.new_empty((0, 3))
        context_ids = target_ids.new_empty((0,))
        unknown = target
        unknown_ids = target_ids
        if self.hparams.outpaint:
            unconditional = torch.rand((), device=target.device) < self.hparams.uncond_prob
            if not unconditional:
                known_flags = self._sample_valid_known_flags(target_ids)
                if known_flags.any() and (~known_flags).any():
                    context, context_ids = target[known_flags], target_ids[known_flags]
                    unknown, unknown_ids = target[~known_flags], target_ids[~known_flags]
        else:
            # In the unconditional single-object setting, the frozen Teacher
            # must receive the complete GT cloud as training-only context.
            # Without this branch, the default empty context silently turns a
            # GT-context Teacher into a source-only Teacher during Student
            # distillation.
            context, context_ids = target, target_ids

        # CUDA bincount is not implemented as a deterministic kernel in the
        # supported PyTorch version.  Keep deterministic mode for the rest of
        # the training step, but temporarily allow this bookkeeping operation.
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        except AttributeError:
            warn_only = False
        if deterministic:
            torch.use_deterministic_algorithms(False)
        try:
            counts = torch.bincount(unknown_ids, minlength=self.spec.num_voxels)
        finally:
            if deterministic:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
        source, source_ids = sample_gaussian_anchors(
            self.spec,
            counts,
            self.hparams.anchor_noise_std_ratio,
            clip_to_voxel=self.hparams.anchor_clip_to_voxel,
        )
        source = source.to(dtype=target.dtype)
        return source, source_ids, unknown, unknown_ids, context, context_ids

    def flow_loss(
        self,
        targets,
        layout=None,
        *,
        condition=None,
        layout_coords=None,
        layout_feats=None,
        scene_keys=None,
    ):
        """Return only student velocity MSE against frozen teacher endpoints."""
        if not targets:
            raise ValueError("targets cannot be empty")
        if condition is not None:
            if layout is not None or layout_coords is not None or layout_feats is not None:
                raise ValueError("provide condition or layout, not both")
            layout = condition
        layout = self.prepare_layout_condition(
            layout,
            len(targets),
            targets[0].dtype,
            layout_coords=layout_coords,
            layout_feats=layout_feats,
            dropout=self.training,
        )
        sources = []
        source_ids = []
        source_batch_ids = []
        teacher_targets = []
        teacher_target_ids = []
        teacher_contexts = []
        teacher_context_ids = []
        student_contexts = []
        student_context_ids = []
        student_context_batch_ids = []

        for batch_id, target in enumerate(targets):
            (
                source,
                source_id,
                unknown,
                unknown_id,
                context,
                context_id,
            ) = self._scene_pair(target)
            sources.append(source)
            source_ids.append(source_id)
            source_batch_ids.append(
                torch.full((source.shape[0],), batch_id, dtype=torch.long, device=source.device)
            )
            teacher_targets.append(unknown)
            teacher_target_ids.append(unknown_id)
            teacher_contexts.append(context)
            teacher_context_ids.append(context_id)
            if self.hparams.outpaint:
                student_contexts.append(context)
                student_context_ids.append(context_id)
                student_context_batch_ids.append(
                    torch.full((context.shape[0],), batch_id, dtype=torch.long, device=context.device)
                )

        with torch.no_grad():
            # The installed spconv build supports this large-scene frozen
            # Teacher path in FP32; Student training still follows Lightning
            # AMP FP16 below. Sinkhorn keeps internal reductions in FP32.
            with torch.cuda.amp.autocast(enabled=False):
                endpoints = self.mapping_teacher.map_batch(
                    sources,
                    source_ids,
                    teacher_targets,
                    teacher_target_ids,
                    teacher_contexts,
                    teacher_context_ids,
                )

                refined_endpoints = []
                refiner_diagnostics = []
                for batch_id, (source_item, endpoint_item, target_item) in enumerate(
                    zip(sources, endpoints, teacher_targets)
                ):
                    cache_key = None
                    if scene_keys is not None and batch_id < len(scene_keys):
                        cache_key = str(scene_keys[batch_id])
                    refined_item, diagnostics = self.endpoint_refiner.refine(
                        source_item,
                        endpoint_item,
                        target_item,
                        cache_key=cache_key,
                    )
                    refined_endpoints.append(refined_item)
                    refiner_diagnostics.append(diagnostics)
                endpoints = refined_endpoints
                self._last_endpoint_refiner_diagnostics = refiner_diagnostics

        # Teacher and sparse OT are inference-only for the Student.  Release
        # their cached sparse-convolution/KNN workspaces before constructing
        # the Student state and running PTv3, while keeping only the compact
        # source-indexed refined endpoints needed below.
        if source[0].is_cuda and self.hparams.clear_teacher_cache:
            torch.cuda.synchronize(source[0].device)
            gc.collect()
            torch.cuda.empty_cache()

        source = torch.cat(sources, dim=0)
        source_ids_flat = torch.cat(source_ids, dim=0)
        source_batch_flat = torch.cat(source_batch_ids, dim=0)
        endpoint = torch.cat(endpoints, dim=0)
        times = torch.rand((len(targets),), device=source.device, dtype=source.dtype)
        point_times = times[source_batch_flat].unsqueeze(1)
        state = (1.0 - point_times) * source + point_times * endpoint
        target_velocity = endpoint - source
        centers = self.spec.voxel_centers(source_ids_flat).to(source.dtype)
        source_layout = self._layout_features(layout, source_batch_flat, source_ids_flat)

        if self.hparams.outpaint:
            context = torch.cat(student_contexts, dim=0)
            context_ids = torch.cat(student_context_ids, dim=0)
            context_batch = torch.cat(student_context_batch_ids, dim=0)
            context_centers = self.spec.voxel_centers(context_ids).to(source.dtype)
            network_points = torch.cat([state, context], dim=0)
            network_centers = torch.cat([centers, context_centers], dim=0)
            network_batch = torch.cat([source_batch_flat, context_batch], dim=0)
            known_flags = torch.cat(
                [
                    torch.zeros(source.shape[0], dtype=torch.bool, device=source.device),
                    torch.ones(context.shape[0], dtype=torch.bool, device=source.device),
                ]
            )
            context_layout = self._layout_features(layout, context_batch, context_ids)
            network_layout = (
                torch.cat([source_layout, context_layout], dim=0)
                if source_layout is not None
                else None
            )
            prediction = self.net.forward_conditional(
                network_points,
                network_centers,
                times,
                batch_ids=network_batch,
                known_flags=known_flags,
                extra_features=network_layout,
            )[: source.shape[0]]
        else:
            prediction = self.net(
                state,
                centers,
                times,
                batch_ids=source_batch_flat,
                extra_features=source_layout,
            )
        return F.mse_loss(prediction, target_velocity)

    def training_step(self, batch, _):
        targets = [points.to(self.device) for points in batch["points"]]
        condition = batch.get("condition")
        if condition is None and self.hparams.condition_type == "semantic_kitti_lidar":
            from pointworldmodel.core.data.semantic_kitti import (
                semantic_kitti_condition_from_batch,
            )

            condition = semantic_kitti_condition_from_batch(
                batch,
                self.spec,
                device=self.device,
                dtype=targets[0].dtype,
                count_upper_bound=float(self.hparams.condition_count_upper_bound),
            )
        loss = self.flow_loss(
            targets,
            batch.get("layout"),
            condition=condition,
            layout_coords=batch.get("layout_coords"),
            layout_feats=batch.get("layout_feats"),
            scene_keys=batch.get("path"),
        )
        self.log("train/point_flow_mse", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            self,
            lr=self.hparams.lr,
            optimization=self.hparams.optimization,
        )

    @torch.no_grad()
    def sample(
        self,
        source,
        source_ids,
        steps=16,
        known_points=None,
        known_ids=None,
        layout=None,
        condition=None,
        layout_coords=None,
        layout_feats=None,
    ):
        """Integrate the student ODE from Gaussian anchors to point samples."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if source.ndim != 2 or tuple(source.shape[1:]) != (3,):
            raise ValueError("source must have shape [N, 3]")
        if source.device != self.device:
            raise ValueError("source must be on the same device as PointFlow")
        if source_ids.ndim != 1 or source_ids.numel() != source.shape[0]:
            raise ValueError("source_ids must have shape [len(source)]")
        if (source_ids < 0).any() or (source_ids >= self.spec.num_voxels).any():
            raise ValueError("source_ids contains invalid voxel IDs")
        if self.hparams.outpaint and ((known_points is None) != (known_ids is None)):
            raise ValueError("known_points and known_ids must be provided together")
        if not self.hparams.outpaint and known_points is not None:
            raise ValueError("this checkpoint was not trained for outpainting")

        if condition is not None:
            if layout is not None or layout_coords is not None or layout_feats is not None:
                raise ValueError("provide condition or layout, not both")
            layout = condition

        x = source
        source_ids = source_ids.to(device=x.device, dtype=torch.long)
        layout = self.prepare_layout_condition(
            layout,
            1,
            x.dtype,
            layout_coords=layout_coords,
            layout_feats=layout_feats,
        )
        source_batch = torch.zeros(source_ids.shape[0], dtype=torch.long, device=x.device)
        centers = self.spec.voxel_centers(source_ids).to(x.dtype)
        source_layout = self._layout_features(layout, source_batch, source_ids)

        if self.hparams.outpaint:
            if known_points is None:
                known_points = x.new_empty((0, 3))
                known_ids = source_ids.new_empty((0,))
            else:
                known_points = known_points.to(device=x.device, dtype=x.dtype)
                known_ids = known_ids.to(device=x.device, dtype=torch.long)
                if known_points.ndim != 2 or tuple(known_points.shape[1:]) != (3,):
                    raise ValueError("known_points must have shape [N, 3]")
                if known_ids.ndim != 1 or known_ids.numel() != known_points.shape[0]:
                    raise ValueError("known_ids must have shape [len(known_points)]")
                if (known_ids < 0).any() or (known_ids >= self.spec.num_voxels).any():
                    raise ValueError("known_ids contains invalid voxel IDs")
            known_batch = torch.zeros(known_ids.shape[0], dtype=torch.long, device=x.device)
            known_centers = self.spec.voxel_centers(known_ids).to(x.dtype)
            known_layout = self._layout_features(layout, known_batch, known_ids)
            network_layout = (
                torch.cat([source_layout, known_layout], dim=0)
                if source_layout is not None
                else None
            )
            known_flags = torch.cat(
                [
                    torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
                    torch.ones(known_points.shape[0], dtype=torch.bool, device=x.device),
                ]
            )

        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((1,), step / steps, device=x.device, dtype=x.dtype)
            if self.hparams.outpaint:
                prediction = self.net.forward_conditional(
                    torch.cat([x, known_points], dim=0),
                    torch.cat([centers, known_centers], dim=0),
                    time,
                    batch_ids=torch.cat([source_batch, known_batch], dim=0),
                    known_flags=known_flags,
                    extra_features=network_layout,
                )[: x.shape[0]]
            else:
                prediction = self.net(
                    x,
                    centers,
                    time,
                    batch_ids=source_batch,
                    extra_features=source_layout,
                )
            x = x + dt * prediction
        return x

    @torch.no_grad()
    def sample_batch(self, source, source_ids, batch_ids, steps=16):
        """Integrate several unconditional scenes in one PTv3 forward batch.

        ``source`` is a concatenation of variable-length scenes and
        ``batch_ids`` identifies the scene owning each point.  The official
        PTv3 adapter already uses this representation for training; exposing
        the same layout here avoids reloading/rerunning the network once per
        generated shape during evaluation.

        Outpainting is intentionally not handled by this minimal path yet:
        the chair validation experiment is unconditional, and conditional
        callers should continue using :meth:`sample` with explicit context.
        """
        if self.hparams.outpaint:
            raise NotImplementedError("sample_batch currently supports unconditional PointFlow only")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if source.ndim != 2 or tuple(source.shape[1:]) != (3,):
            raise ValueError("source must have shape [N, 3]")
        if source_ids.ndim != 1 or source_ids.numel() != source.shape[0]:
            raise ValueError("source_ids must have shape [len(source)]")
        if batch_ids.ndim != 1 or batch_ids.numel() != source.shape[0]:
            raise ValueError("batch_ids must have shape [len(source)]")
        if source.numel() == 0:
            raise ValueError("source must contain at least one point")
        batch_ids = batch_ids.to(device=source.device, dtype=torch.long).reshape(-1)
        if (batch_ids < 0).any():
            raise ValueError("batch_ids must be non-negative")
        batch_count = int(batch_ids.max().item()) + 1
        if set(torch.unique(batch_ids).detach().cpu().tolist()) != set(range(batch_count)):
            raise ValueError("batch_ids must contain contiguous scene IDs starting at zero")
        source_ids = source_ids.to(device=source.device, dtype=torch.long)
        if (source_ids < 0).any() or (source_ids >= self.spec.num_voxels).any():
            raise ValueError("source_ids contains invalid voxel IDs")

        x = source
        centers = self.spec.voxel_centers(source_ids).to(dtype=x.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((batch_count,), step / steps, device=x.device, dtype=x.dtype)
            prediction = self.net(
                x,
                centers,
                time,
                batch_ids=batch_ids,
            )
            x = x + dt * prediction
        return x
