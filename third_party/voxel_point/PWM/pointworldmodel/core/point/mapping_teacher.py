"""Chamfer-only source-to-GT transport teacher.

This module is the voxel PointFlow analogue of PWM's teacher stage.  The
teacher is used only while training the student:

* source points are sampled from the same voxel counts as the target;
* source and GT points are presented together, with GT marked as context;
* the network predicts a displacement for each source point;
* the predicted source-indexed endpoint is supervised only by symmetric
  Chamfer distance.

There is deliberately no Sinkhorn target, pointwise MSE, kNN repulsion, TV,
or other auxiliary objective in this module.  A student can consume the
resulting endpoint without receiving GT at inference time.
"""

from __future__ import annotations

import gc
from typing import List, Sequence

import torch
from pytorch_lightning import LightningModule

from pointworldmodel.core.point.anchor import sample_gaussian_anchors
from pointworldmodel.core.voxel.grid import VoxelSpec
from pointworldmodel.core.conditioning.outpaint import point_known_flags, random_known_masks
from pointworldmodel.core.point.ptv3_active_network import OfficialPTV3PointFlowNet
from pointworldmodel.core.runtime.optimization import build_optimizer_and_scheduler


_CHAMFER = None


def _symmetric_sqrt_chamfer(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return PWM's bidirectional square-root Chamfer distance.

    The pinned CUDA implementation is used for normal training.  The
    ``torch.cdist`` fallback keeps small CPU smoke tests usable when the
    optional extension is unavailable.
    """
    if predicted.ndim != 2 or target.ndim != 2:
        raise ValueError("predicted and target must have shape [N, 3]")
    if predicted.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError("Chamfer inputs must contain XYZ coordinates")
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("Chamfer inputs must be non-empty")

    global _CHAMFER
    # Never JIT-build a CUDA extension for CPU validation.  Apart from being
    # unnecessary, torch's extension loader may fail on machines without a
    # visible CUDA architecture even though the differentiable CPU fallback
    # is perfectly adequate for small smoke tests.
    if _CHAMFER is None and predicted.is_cuda:
        try:
            from pointworldmodel.models.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import (
                chamfer_3DDist,
            )

            _CHAMFER = chamfer_3DDist()
        except Exception:
            _CHAMFER = False

    # The pinned extension is CUDA-only even when its Python module can be
    # imported on CPU.  Use the differentiable torch fallback for CPU smoke
    # tests instead of letting the extension call ``torch.cuda.set_device``.
    if _CHAMFER is not False and predicted.is_cuda:
        dist_pred, dist_target, _, _ = _CHAMFER(
            predicted.unsqueeze(0).contiguous(),
            target.unsqueeze(0).contiguous(),
        )
        return 0.5 * (
            torch.sqrt(dist_pred.clamp_min(0)).mean()
            + torch.sqrt(dist_target.clamp_min(0)).mean()
        )

    if predicted.is_cuda and max(predicted.shape[0], target.shape[0]) > 8192:
        raise RuntimeError(
            "The CUDA ChamferDistancePytorch extension is required for "
            "large GPU teacher batches; the torch.cdist fallback is only "
            "intended for small CPU smoke tests."
        )

    distances = torch.cdist(predicted, target).square()
    return 0.5 * (
        torch.sqrt(distances.min(dim=1).values.clamp_min(0)).mean()
        + torch.sqrt(distances.min(dim=0).values.clamp_min(0)).mean()
    )


class PointMappingTeacher(LightningModule):
    """Learn a source-indexed endpoint using GT as training-only context.

    ``OfficialPTV3PointFlowNet`` normally receives six channels per point: local XYZ
    and voxel-centre XYZ.  The teacher uses a seventh ``known`` channel.  A
    source point has flag 0 and a GT context point has flag 1.  The network
    processes both sets jointly, but only the source outputs are converted to
    endpoints.  This preserves the source ordering required by a downstream
    flow-matching student.
    """

    def __init__(
        self,
        spec=None,
        spec_config=None,
        lr=2e-4,
        optimization=None,
        base_channels=32,
        quantization_size=0.1,
        bottleneck_blocks=2,
        backbone="ptv3",
        fine_token_size=0.05,
        coarse_token_size=0.10,
        intra_voxel_encoding_bands=4,
        fine_dim=None,
        coarse_dim=None,
        fine_depth=3,
        coarse_depth=4,
        decoder_depth=2,
        clear_cuda_cache=True,
        fine_heads=8,
        coarse_heads=8,
        fine_window_size=64,
        coarse_window_size=128,
        dropout=0.0,
        anchor_noise_std_ratio=0.25,
        anchor_clip_to_voxel=False,
        outpaint=False,
        mask_min_fraction=0.25,
        mask_max_fraction=0.50,
        mask_modes=("directional", "corner"),
        uncond_prob=0.20,
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
        if backbone not in {"ptv3", "ptv3_official"}:
            raise ValueError(
                "PointMappingTeacher currently supports only the official "
                "PTv3 backbone ('ptv3_official')"
            )
        if anchor_noise_std_ratio < 0:
            raise ValueError("anchor_noise_std_ratio must be non-negative")
        if not 0.0 <= uncond_prob <= 1.0:
            raise ValueError("uncond_prob must lie in [0, 1]")

        spec_config = spec.to_dict()
        self.save_hyperparameters(ignore=["spec"])
        self.spec = spec
        coordinate_scale = (
            (spec.x_max - spec.x_min) / 2.0,
            (spec.y_max - spec.y_min) / 2.0,
            (spec.z_max - spec.z_min) / 2.0,
        )
        # Teacher and student must use the exact same PTv3 implementation.
        self.net = OfficialPTV3PointFlowNet(
            in_channels=7,
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


    def _sample_pair(self, target: torch.Tensor):
        """Create anchors and the GT context seen by the teacher.

        For the unconditional single-object teacher, the complete GT cloud is
        always provided as context.  The source anchors remain the only points
        whose predicted displacements are returned and supervised.  The
        outpainting branch keeps its partial-known/unknown split because only
        the unknown region is the transport target there.
        """
        target_ids, valid = self.spec.point_to_indices(target)
        target = target[valid]
        target_ids = target_ids[valid]
        if target.numel() == 0:
            raise ValueError("A teacher target contains no points inside the voxel grid")

        if self.hparams.outpaint:
            context = target.new_empty((0, 3))
            context_ids = target_ids.new_empty((0,))
            known_mask = random_known_masks(
                1,
                self.spec,
                target.device,
                self.hparams.mask_min_fraction,
                self.hparams.mask_max_fraction,
                self.hparams.mask_modes,
            )
            if torch.rand((), device=target.device) >= self.hparams.uncond_prob:
                known_flags = point_known_flags(target_ids, known_mask)
                # If a small object happens to fall completely inside one
                # region, use the whole scene as the unknown target rather
                # than creating an empty transport problem.
                if known_flags.any() and (~known_flags).any():
                    context, context_ids = target[known_flags], target_ids[known_flags]
                    target, target_ids = target[~known_flags], target_ids[~known_flags]
        else:
            # The unconditional teacher must receive the complete GT cloud as
            # training-only context.  ``known_flag=1`` is added later by
            # predict_displacements(); the source anchors use known_flag=0.
            context = target
            context_ids = target_ids

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        except AttributeError:
            warn_only = False
        if deterministic:
            torch.use_deterministic_algorithms(False)
        try:
            counts = torch.bincount(target_ids, minlength=self.spec.num_voxels)
        finally:
            if deterministic:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
        source, source_ids = sample_gaussian_anchors(
            self.spec,
            counts,
            self.hparams.anchor_noise_std_ratio,
            clip_to_voxel=self.hparams.anchor_clip_to_voxel,
        )
        return source, source_ids, target, target_ids, context, context_ids

    def predict_displacements(
        self,
        sources: torch.Tensor,
        source_ids: torch.Tensor,
        targets: torch.Tensor,
        target_ids: torch.Tensor,
        contexts: torch.Tensor,
        context_ids: torch.Tensor,
        source_batch_ids: torch.Tensor,
        target_batch_ids: torch.Tensor,
        context_batch_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Predict source-indexed displacements for a flattened scene batch."""
        if sources.shape[-1] != 3 or targets.shape[-1] != 3:
            raise ValueError("sources and targets must contain XYZ coordinates")
        all_points = torch.cat([sources, contexts], dim=0)
        all_voxel_ids = torch.cat([source_ids, context_ids], dim=0)
        all_batch_ids = torch.cat([source_batch_ids, context_batch_ids], dim=0)
        all_centers = self.spec.voxel_centers(all_voxel_ids).to(all_points.dtype)
        known_flags = torch.cat(
            [
                torch.zeros(sources.shape[0], dtype=torch.bool, device=sources.device),
                torch.ones(contexts.shape[0], dtype=torch.bool, device=contexts.device),
            ]
        )
        # The teacher has no flow-time variable.  Use t=0 as a fixed
        # conditioning value; the student receives the actual flow time.
        zero_time = torch.zeros(batch_size, device=sources.device, dtype=sources.dtype)
        displacement = self.net.forward_conditional(
            all_points,
            all_centers,
            zero_time,
            batch_ids=all_batch_ids,
            known_flags=known_flags,
        )
        return displacement[: sources.shape[0]]

    def map_batch(
        self,
        sources: Sequence[torch.Tensor],
        source_ids: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        target_ids: Sequence[torch.Tensor],
        contexts: Sequence[torch.Tensor] | None = None,
        context_ids: Sequence[torch.Tensor] | None = None,
    ) -> List[torch.Tensor]:
        """Map a list of scene sources while preserving per-scene ordering."""
        if not (len(sources) == len(source_ids) == len(targets) == len(target_ids)):
            raise ValueError("source and target scene lists must have equal length")
        if not sources:
            return []
        if contexts is None:
            contexts = targets
        if context_ids is None:
            context_ids = target_ids
        if len(contexts) != len(sources) or len(context_ids) != len(sources):
            raise ValueError("context scene lists must match source scene lists")
        for index, (source, source_id, target, target_id) in enumerate(
            zip(sources, source_ids, targets, target_ids)
        ):
            if source.ndim != 2 or tuple(source.shape[1:]) != (3,):
                raise ValueError(f"source scene {index} must have shape [N, 3]")
            if target.ndim != 2 or tuple(target.shape[1:]) != (3,):
                raise ValueError(f"target scene {index} must have shape [N, 3]")
            if source.shape[0] != target.shape[0]:
                raise ValueError(
                    f"source and target scene {index} must have equal point counts"
                )
            if source_id.ndim != 1 or source_id.numel() != source.shape[0]:
                raise ValueError(f"source IDs for scene {index} do not align with points")
            if target_id.ndim != 1 or target_id.numel() != target.shape[0]:
                raise ValueError(f"target IDs for scene {index} do not align with points")

        # The teacher is frozen and receives both source and GT context.  With
        # large 3D-FRONT scenes, concatenating multiple scenes here roughly
        # doubles an already large sparse-convolution activation graph.  Keep
        # the dataloader/student batch at the requested size, but evaluate the
        # teacher scene-by-scene to cap its inference peak memory.
        if len(sources) > 1:
            endpoints = []
            for source, source_id, target, target_id, context, context_id in zip(
                sources, source_ids, targets, target_ids, contexts, context_ids
            ):
                source_batch = torch.zeros(
                    source.shape[0], dtype=torch.long, device=source.device
                )
                target_batch = torch.zeros(
                    target.shape[0], dtype=torch.long, device=target.device
                )
                context_batch = torch.zeros(
                    context.shape[0], dtype=torch.long, device=context.device
                )
                displacement = self.predict_displacements(
                    source,
                    source_id,
                    target,
                    target_id,
                    context,
                    context_id,
                    source_batch,
                    target_batch,
                    context_batch,
                    1,
                )
                endpoints.append(source + displacement)
                del displacement
                if source.is_cuda and self.hparams.clear_cuda_cache:
                    torch.cuda.synchronize(source.device)
                    gc.collect()
                    torch.cuda.empty_cache()
            return endpoints

        source_batch_ids = torch.cat(
            [torch.full((len(points),), i, dtype=torch.long, device=points.device)
             for i, points in enumerate(sources)]
        )
        target_batch_ids = torch.cat(
            [torch.full((len(points),), i, dtype=torch.long, device=points.device)
             for i, points in enumerate(targets)]
        )
        context_batch_ids = torch.cat(
            [torch.full((len(points),), i, dtype=torch.long, device=points.device)
             for i, points in enumerate(contexts)]
        )
        flat_sources = torch.cat(list(sources), dim=0)
        flat_source_ids = torch.cat(list(source_ids), dim=0)
        flat_targets = torch.cat(list(targets), dim=0)
        flat_target_ids = torch.cat(list(target_ids), dim=0)
        flat_contexts = torch.cat(list(contexts), dim=0)
        flat_context_ids = torch.cat(list(context_ids), dim=0)
        displacement = self.predict_displacements(
            flat_sources,
            flat_source_ids,
            flat_targets,
            flat_target_ids,
            flat_contexts,
            flat_context_ids,
            source_batch_ids,
            target_batch_ids,
            context_batch_ids,
            len(sources),
        )
        endpoints = flat_sources + displacement
        return list(torch.split(endpoints, [len(points) for points in sources]))

    def training_step(self, batch, _):
        if self.device.type != "cuda":
            raise RuntimeError(
                "PointMappingTeacher training requires CUDA because the "
                "large-scale Chamfer objective must run on the GPU."
            )
        targets = [points.to(self.device) for points in batch["points"]]
        sources = []
        source_ids = []
        clean_targets = []
        target_ids = []
        contexts = []
        context_ids = []
        for target in targets:
            source, source_id, clean_target, clean_target_id, context, context_id = self._sample_pair(target)
            sources.append(source)
            source_ids.append(source_id)
            clean_targets.append(clean_target)
            target_ids.append(clean_target_id)
            contexts.append(context)
            context_ids.append(context_id)

        endpoints = self.map_batch(
            sources,
            source_ids,
            clean_targets,
            target_ids,
            contexts,
            context_ids,
        )
        loss_cd = torch.stack(
            [
                _symmetric_sqrt_chamfer(endpoint, target)
                for endpoint, target in zip(endpoints, clean_targets)
            ]
        ).mean()

        # This is intentionally the complete objective: pure teacher CD.
        self.log("train/teacher_cd", loss_cd, prog_bar=True, sync_dist=True)
        return loss_cd

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            self,
            lr=self.hparams.lr,
            optimization=self.hparams.optimization,
        )
