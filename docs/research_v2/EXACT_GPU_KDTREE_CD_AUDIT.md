# Exact GPU KD-tree Chamfer distance

Implemented and GPU-tested in the Stage2 score/GT-parents worktree. Only physical
GPU1 is used for evaluation; formal GPU2/3 training and source identity are unchanged.
No system packages or CUDA installation are modified. Extension build products stay
in `FPSGen/env/voxel_point_cache/gpu_kdtree_v1` and are not source artifacts.

## Definition and algorithm

CD is unchanged: `0.5*(mean Euclidean nearest distance P->GT + GT->P)`, in meters.
Every original, valid raw-map GT point participates. No Poisson resampling, voxel
center replacement of GT, approximate neighbor search, or metric padding is used.

The balanced KD-tree uses cyclic XYZ splits. GPU int64 lexicographic keys combine
segment identity and exactly ordered FP32 coordinate bits. Segmented sorting at each
level yields nested leaf ranges. CUDA builds leaf and internal bounding boxes.
Each query traverses the tree with an explicit bounded stack, visiting the nearer
child first and pruning only branches with conservative lower bounds exceeding the
current nearest distance. There is no fixed search radius, visit budget or early
approximate termination. CPU performs Python orchestration, not spatial indexing
or nearest-neighbor queries for the GPU result. The implementation is evaluation-only.

Input points and boxes are FP32. Query distance arithmetic and means use FP64 to
match CPU reference arithmetic closely. Bounding-box lower bounds use downward
rounding and outward pruning margins. Exact refers to the search, not a promise of
bitwise equivalence across all floating-point libraries and devices.

The CUDA module has no per-batch shared N/M requirement: each scene builds its own
trees at natural lengths. Stage2 model inference still packs four variable-token
scenes. Different scenes are never merged into one nearest-neighbor search.

## Real full-scene numerical evidence

Seven generated GPU cases cover non-power-of-two counts, duplicate/tied points,
planar surfaces, exact hits and distant queries. All passed. Pytest additionally
passed22 tests, including11 actual CUDA KD-tree tests and11 existing selected metric
and inference tests (`CUDA_VISIBLE_DEVICES=1`). No GPU test was counted as a CPU pass.

Full frame08/003900:504083 predicted points and1144476 raw GT points. Every query
in both directions was compared with SciPy `cKDTree.query(eps=0)`: maximum nearest
distance error0.0m in this run. Final CD:

| Backend | CD(m) |
|---|---:|
| CPU exact KD-tree |0.18014397302200310|
| New exact GPU KD-tree |0.18014397302200302|
| Existing CUDA all-pairs |0.18014397310567554|

GPU-tree versus CPU difference8.33e-17m; versus existing CUDA difference8.37e-11m.
The tiny latter discrepancy is consistent with the old FP32 versus new FP64
distance arithmetic. No change to square-root convention or half-factor occurred.

Four additional real scenes use the exact17500-update checkpoint and packedB4 model:

| Frame | Predicted points | Raw GT points | GPU CD build+query seconds | Maximum per-query CPU error(m) |
|---|---:|---:|---:|---:|
|000000|835526|1728347|0.231831|0|
|000100|821565|2381384|0.264710|0|
|001800|843202|2450855|0.255248|0|
|004000|816495|1126190|0.158987|0|

Against the previous CUDA metric results on identical point sets, maximum CD
difference1.044e-10m. All IoU TP/FP/FN integers agree exactly. IoU code is unchanged.
The reported zero per-query errors describe these tested inputs, not a universal
claim of bitwise equality on every possible point cloud.

## Timing

For003900, full-shape warm GPU-tree runs including BOTH fresh tree builds, BOTH
queries and reductions were0.138677,0.133127,0.134571s (median0.134571s).
The same-process existing all-pairs CUDA CD took5.375497s (about40x this median).
Earlier repeated CPU exact KD-tree timings on identical points had median1.720167s
(about12.8x). Timing is workload/hardware dependent and is CD-only, excluding model,
raw-map preprocessing and transfer. Compilation70.71s is startup, separately recorded.
This does not imply the complete inference pipeline is40x faster.

The four-scene GPU-tree CD total is0.910776s. Old CUDA B4 CD+IoU was39.172011s;
that older number includes IoU, so it is not a pure-CD controlled timing comparison.

## Implementation and preserved evaluation

Implementation: `fpsgen/ops/gpu_kdtree.py`, `gpu_kdtree_cuda.cu`, and
`fpsgen/utils/gpu_kdtree_metrics.py`. Tests: `tests/test_gpu_kdtree.py`.
Audit/benchmark scripts are under `scripts/research_v2/hier_occ_inference/`.

The old CUDA evaluation was safely paused via SIGINT at64/408 frames, retaining its
sources, journals, protocol and checkpoint. New evaluation uses a separate output:
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/morph_step017500_seq08_skip10_s1nfe4_s2nfe4_batch4_gpu_kdtree_gpu1`.
It restarts the408-frame metric evaluation rather than silently mixing backends.
Checkpoint, seeds, Stage1/Stage2 NFE4/4, support thresholds, raw GT and physical0.5m
IoU are unchanged. Metric backend identity, source and compiled-module hashes are
explicit in the frozen job. Final report is generated only after408 valid rows.

tmux: `fpsgen_hs_morph17500_seq08_skip10_kdtree_b4_gpu1`.

Evidence JSONs in `docs/research_v2/stage2_morph_fm100k_audit/`:
`exact_gpu_kdtree_full_rawmap_audit.json`, `exact_gpu_kdtree_four_scene_audit.json`,
`exact_gpu_kdtree_old_cuda_parity.json`. Full408 metrics are pending at launch.

Launch verified: PID3182960 on GPU1,9314MiB observed, same checkpoint17500.
Initial launch correctly refused a concurrent GPU1 pytest job; it was not stopped
and naturally finished. Explicit metric-evaluation resume then passed admission.
See `stage2_morph_fm100k_audit/gpu_kdtree_launch_verified.json`.
