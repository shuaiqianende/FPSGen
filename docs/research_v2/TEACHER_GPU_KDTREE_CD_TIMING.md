# Mapping Teacher180K: differentiable exact GPU KD-tree CD benchmark

Measured on physicalGPU1 only, single scene00/000000. Teacher training mode,FP16 AMP, activation checkpointing.
Source180000 + GT context180000 =360000 network input points; source endpoint has180000 points.
Original14,708,739-parameter Teacher final50K checkpoint was read only for representative endpoint profiling.
No optimizer step, checkpoint mutation, training restart or GPU2/3 task modification occurred.

## Historical backend

The completed100K-frame GT-source Teacher run used `runtime.knn_backend: keops`, not `chamfer_3D`.
`train_core.py` calls `voxel_point.metrics.chamfer_sqrt`: KeOps returns nearest indices under no-grad,
then PyTorch recomputes selected coordinate norms for differentiable bidirectional square-root CD.
Historical logs provide whole-step mean1.441758s, including training/DDP/I/O overhead; they do not split CD/kernel/network timing.

## Measurement

Three repeated full-size warm trials per backend after a separate warmup. CUDA is synchronized around each timed section.
Below are independent medians, which need not sum exactly to the median total. The total excludes data loading,
optimizer update, gradient-finiteness scans, DDP communication and checkpointing. It is forward+backward compute,
not a measured formal optimizer-update throughput. Single-frame result is not a full-dataset speed guarantee.

| CD backend | Teacher forward(ms) | CD forward incl. search/build(ms) | CD backward(ms) | Teacher backward(ms) | Total(ms) |
|---|---:|---:|---:|---:|---:|
|keops|305.707|250.931|1.024|729.650|1293.492|
|chamfer_cuda|281.153|313.998|1.320|739.656|1335.792|
|gpu_kdtree|305.818|19.770|1.494|743.939|1070.299|

New KD-tree reduces measured compute time17.26% versus KeOps (throughput factor1.209x).
Most remaining time is the network backward. First KeOps code generation21.15s and first Teacher backward13.11s
are separately preserved as warmup and excluded from medians. Original CUDA timing fluctuates across phases;
raw repetitions are retained, not replaced by the best run. Peak allocated memory8.857GiB.

## Loss and gradients

All three backends loss0.035387761890888214 on the identical Teacher endpoint.
New KD-tree versus KeOps endpoint gradient relativeL2=0 and max absolute difference=0 on this sample.
Original CUDA versus KeOps relativeL2=6.153e-8 and max absolute difference1.819e-12.
All full network parameter gradients finite. This is tested numerical agreement, not a universal bitwise guarantee.
Nearest-neighbor ties or floating-point near-ties can give equivalent distances with different valid subgradients.

The optional adapter `fpsgen/utils/gpu_kdtree_training.py` uses the exact CUDA tree only for discrete nearest indices;
PyTorch gather/norm supplies coordinate derivatives, including finite zero-distance behavior.
It shares the existing KeOps loss convention and0.5 factor. The evaluation-only tree output itself is not an autograd loss.
No production Teacher loss/backend was changed automatically.

Evidence: `stage2_morph_fm100k_audit/teacher180k_cd_forward_backward_gpu1.json`.
GPU tests additionally compare prediction AND target gradients to a small dense oracle and cover exact zero distances.
