# VoxDet Stage1 ROI filtering and throughput restart — 2026-10-13

The initial formal GPU2+3 run stopped at223 successful updates at07:55:29
local time. GT00/000447 contained one float32 XYZ with XY radius
50.00000026056325m, outside the strict open50m disk. No checkpoint milestone
had been reached. The user explicitly authorized filtering these out-of-ROI
points and requested higher GPU memory utilization to improve throughput.

Stage1 now opts into `data.gt_roi_policy: filter`. Raw GT must still contain
exactly180000 finite XYZ. Out-of-ROI rows are removed, and integer voxel counts
sum to the actual retained cardinality. Density mass loss and density evaluation
use this actual mass. Original/retained/filtered counts are logged per scene.
GT files are unchanged; no point duplication or replacement occurs. The
generated anchor/final-output contract remains180000 points. Dataset callers
without the explicit filter policy retain strict validation. Other stage
configurations are not silently changed by this Stage1 update.

The official pinned tokenizer accepts only one scene at a time. The adapter
therefore tokenizes variable-size scenes separately and concatenates their
dense coarse-grid tensors before the existing3D trunk, neck and heads. Upstream
source, geometry, all widths/depths/refinement layers and22,082,470 parameters
are preserved. Batching changes training batch statistics. No padding,
downsampling of scenes, hidden cross-stage features or semantic condition is
introduced.

Measurements use31 real training scenes drawn from all ten train sequences,
including the failing frame; no seq08. Single-GPU warm timings exclude the
first update; DDP uses32 successful updates and reports the slower rank.

| Per-GPU batch | Activation checkpointing | CPU workers/rank | Warm s/update | Frames/s/rank | Peak allocated GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | off | 0 | 0.630 | 1.587 | 9.638 |
| 2 | on | 0 | 1.401 | 1.427 | 9.230 |
| 4 | on | 0 | 2.496 | 1.603 | 18.189 |
| 2 | off | 0 | 1.087 | 1.839 | 18.965 |
| 2 | off | 2 | 0.970 | 2.062 | 18.965 |
| **2, two-GPU DDP** | **off** | **2** | **1.015** | **1.970** | **19.049** |

All gradients are finite. DDP peak reserved memory20.061GiB/rank, leaving
headroom on24GiB RTX3090 cards for CUDA context and input-size variation.
Aggregate DDP throughput is approximately3.94frames/s. Exact timings, filtered
counts and effective config hashes are in
`voxel_point_reports/roi_filter_batch_tuning.json`; these are throughput and
correctness results, not trained density-quality claims.20 CPU tests pass.

The selected formal setting is physical GPUs2/3 only, B2/GPU, FP16 AMP,
no activation recomputation, two data workers/rank with pinned prefetch.
The train manifest is frozen from the original17870-train/900-development split;
training order is reproducibly shuffled per epoch with rank-disjoint batches.
AdamW/OneCycleLR max LR0.0003 and loss weights are unchanged. Global batch4
requires25000 successful updates for100000 scene exposures; the original B1
required50000 updates. This is a restart from initialization, because the failed
run has no saved weights. No Teacher/Sinkhorn/Student long queue is started.

An initial checkpoint is saved at100 successful updates, then every1000
updates, plus the25/50/75/100% milestones. Writes use a temporary file followed
by atomic rename. Each run directory includes frozen config/geometry/manifests,
launch PID/command, Git HEAD and SHA256 provenance, preflight and per-rank JSONL
logs. Large artifacts remain ignored by Git.
