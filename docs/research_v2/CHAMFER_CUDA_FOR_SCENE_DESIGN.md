# Chamfer CUDA for Scene

Implemented as independent `fpsgen/ops/chamfer_scene.py` and
`fpsgen/ops/chamfer_scene_cuda.cu`. Original `fpsgen/ops/chamfer.py`, upstream
`chamfer3D.cu` and `chamfer_cuda.cpp` are untouched; SHA256 checks passed before
and after benchmarks. Original experiments and their checkpoint identities are
preserved. Development/tests use physicalGPU1; GPU2/3 training is not changed.

## Why a separate scene implementation

The local upstream README benchmarks batch32 clouds with1000/2000 points. It does
not say the library supports only objects or has a hard8192-point ceiling.
The actual pinned forward launch uses `dim3(32,16)` with the first dimension
iterating scenes. For B1 only16 query-axis blocks perform distance work, while
each query still scans every support point. The original kernel also revisits
global query/result data across support tiles. This explains concrete scene
workload inefficiencies beyond generic statements about GPU versus CPU.

Two changes are independently measurable:

1. `mode="tiled"`: query-sized dynamic grid,shared-memory SoA support tiles,
   register-resident minimum throughout support scanning. Still exact all-pairs
   O(N*M) compute, without allocating anN*M matrix. This isolates CUDA kernel
   optimization from changes to the nearest-neighbor algorithm.
2. `mode="bvh"`: one GPU Morton sort per cloud, balanced leaf hierarchy,CUDA
   bounding boxes and conservative exact branch-and-bound queries. Query sorting
   improves spatial coherence; results restore original point ordering.
   Minimum-original-index metadata accelerates exact zero-distance duplicate
   ties. No sampling,radius cap,visit budget or approximate KNN.

Both have a separate native CUDA coordinate backward using saved original nearest
indices. FP32 squared distances match the original precision class. This differs
from the earlier KD-tree's FP64 search arithmetic and therefore is not a
controlled experiment isolating only the indexing structure.

## API and build

```python
from fpsgen.ops.chamfer_scene import SceneChamferDistance

metric = SceneChamferDistance(mode="auto")
loss = metric.sqrt_loss(predicted.float(), target.float())
loss.backward()

# Or obtain raw squared distances and original-order nearest IDs:
dp2, dg2, pred_to_gt_ids, gt_to_pred_ids = metric(predicted, target)
```

Inputs may be `[N,3]`/`[M,3]`, dense `[B,N,3]`/`[B,M,3]`, or equal-length lists
of variable-size scenes. Dense/list scenes are independently dispatched, not
merged into a common support. No padding; indices are int64 (original wrapper
returns int32). `sqrt_loss` uses0.5 times the two directional mean Euclidean
distances. Ragged scenes receive equal scene weights. Raw-distance users retain
control of their reduction convention.

`auto` uses tiled when N*M<=2^30, otherwise BVH. This conservative size heuristic
is configurable via `brute_force_pair_limit`; it is not a universally tuned
distribution-aware dispatch rule. Explicit modes remain available. Neither path
depends on KeOps or the original Chamfer binary. Required build tools are the
existing PyTorch/CUDA/Ninja environment. Compilation stays inside
`FPSGen/env/voxel_point_cache/chamfer_scene_v1`.

Coordinate inputs are CUDA FP32 and finite. Distances use original world
coordinates; Morton normalization is only an index-construction detail. No ROI
or semantic interpretation is built into the operator. Sqrt zero matches use a
finite zero coordinate derivative. The native custom backward is first-order
only. Atomic accumulation is not guaranteed bitwise deterministic. Extremely
large finite coordinates can overflow FP32 squared distances; caller training
must continue rejecting nonfinite loss, as with the original library. A boundary
test ensures this cannot return an invalid nearest index.

## Measured scene-scale results

See `CHAMFER_CUDA_FOR_SCENE_SCALING.md` for all11 sizes,5 backends,55 combinations
and165 measured trials. Uniform synthetic XYZ is used; each backend receives
identical full clouds, fresh spatial indices, full-shape warmup and three trials.
Numbers below are synchronized median forward+backward wall milliseconds:

| Points per cloud | Original CUDA | Scene tiled | Scene BVH | Prior GPU KD-tree |
|---:|---:|---:|---:|---:|
|8192|1.922|2.317|3.227|7.936|
|65536|54.822|14.254|5.548|16.105|
|262144|646.569|141.325|12.846|49.214|
|1048576|10318.401|2229.286|52.774|136.295|

At1048576x1048576, the tiled-only improvement is4.63x; BVH adds an algorithmic
pruning gain. Median extra live PyTorch allocated memory at that size is48.001MiB
original,64.001MiB tiled,70.276MiB BVH,124.276MiB prior KD-tree. Inputs and network
memory are excluded; raw non-Torch allocations may not appear in these counters.
Original runtime min/max varies strongly and is retained; do not interpret a
single median ratio as a universal hardware/point-distribution speedup.

All55 scalar CD values match CPU exact KD-tree within1e-5m; largest actual error
1.768e-7m. All165 measured gradients finite. This scalar check alone does not
prove gradient equality.

## Real Teacher training-compute probe

Real18w source +108w raw-map GT context,126w total network input,18w endpoint.
Same14,708,739-parameter historical Teacher read only for profiling. A profiling
adapter relaxes the legacy context-length guard; PTv3 parameters and architecture
are unchanged. No optimizer update or formal training continuation occurred.

On the SAME fixed Teacher endpoint, loss0.06746291369199753 for every backend.
Scene BVH versus KeOps gradient relativeL2=8.704e-8,max absolute1.819e-12; Scene
tiled relativeL2=8.703e-8. Both pass the pre-existing1e-4 relative tolerance.
Full parameter gradients are finite.

| Backend | CD forward(ms) | Teacher forward+CD+all backward(s) |
|---|---:|---:|
|KeOps|1436.365|3.481917|
|Original CUDA|1259.651|3.366997|
|Scene tiled|408.149|2.463609|
|Scene BVH|13.294|2.062467|
|Prior GPU KD-tree|67.912|2.121858|

Independent component medians need not sum to total median. The complete compute
reduction versus KeOps is40.77%, not the CD-only speed ratio. DDP,I/O,optimizer
and checkpoint overhead are excluded. Network backward remains about1.45s.

This run's initial Teacher endpoint is slightly different from the earlier probe
because separate sparse-network forwards are not bitwise deterministic despite
seed resets. The old KD-tree also passed on THIS new endpoint. Earlier Teacher
gradient-gate failures remain preserved and are not relabeled solved. This is a
successful sample-level check, not universal subgradient/convergence equivalence.

## Tested limits and acceptance boundary

- Small64x64 clouds: original0.208ms,Scene tiled0.306ms,Scene BVH0.576ms.
- Small1024x8192 random clouds: original0.431ms,tiled0.462ms,BVH1.293ms.
- Identical zero-distance1024x8192 points: original0.452ms,BVH0.660ms,prior
  KD-tree11.501ms. Zero-tie pruning improves the old KD failure case.
- Identical support but NONZERO equidistant queries: original0.383ms,BVH2.297ms.
  Spatial pruning can still weaken; this is not an all-distribution winner.
- Closest-neighbor ties/rounding can change valid subgradients. Long-training
  convergence and multi-GPU integration are not established by these probes.
- 50 tests pass:26 Scene GPU cases plus24 existing KD/metric/inference regressions.
  Scene tests cover distances/IDs,both coordinate gradients,zero/tied points,
  ragged/dense batches,noncontiguous input,current CUDA stream and FP32 overflow
  index validity. No GPU pass is inferred from a CPU-only test.

The current HS-OccFlow Stage2 trains pure FM without a CD loss. No training backend
was switched. The new library is an independent choice for metrics and future
point-Teacher work after appropriate training checks.

## Provenance and reproducibility

Raw experiments: `FPSGen/experiments/hs_stage2_eval/chamfer_scene_scaling_1024_to_1048576_gpu1`,
`teacher180k_1080k_scene_cuda_gpu1.json`,
`chamfer_scene_small_and_duplicate_limits_gpu1.json` and
`chamfer_scene_final_release_gpu1.json`.
Small CSV/JSON/PNG evidence is in `docs/research_v2/stage2_morph_fm100k_audit`.

The full55-case scaling source snapshot predates a one-line boundary fix that
initializes a valid tiled index when ALL squared distances overflow FP32. Normal
finite-distance computation is unchanged. Final50-test regression and six-case
release smoke at8192,65536,and1048576 for BOTH paths validate the final source and
record final source/binary hashes. Historical timings are not silently rewritten.

The original ChamferDistancePytorch MIT attribution and source provenance remain
in place; the new source is FPSGen-owned independent implementation. No system
dependency,CUDA installation,driver or global configuration was modified.
