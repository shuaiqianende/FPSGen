# Chamfer CUDA for Scene: independent implementation and scaling

COMPLETE:55 combinations,165 measured trials on physicalGPU1 /RTX3090. All original Chamfer source hashes unchanged.
Original upstream README benchmarks B32x1000/2000; it does not impose an8192-point limit or prove an object-only design.
Inspection shows the original dim3(32,16) grid activates only the16 query-axis blocks for B1, and scans all support points.
This is evidence of a scene workload mismatch, not a claim that CUDA itself should lose to CPU.

## Independent implementation

`fpsgen/ops/chamfer_scene.py` and `chamfer_scene_cuda.cu` define a separate extension/cache, not a patch to chamfer_3D.
- tiled: query-sized dynamic CUDA grid; supports staged in shared SoA tiles; keep nearest minimum in registers through all tiles.
- bvh: single GPU Morton sort per cloud, balanced leaf hierarchy, conservative bounding-box pruning, spatially ordered queries, original-order distance/index restoration.
- CUDA coordinate backward uses saved original nearest indices with atomic accumulation; denseB and ragged lists keep scene identities separate.
- Exact search, no point sampling, finite radius, approximate visit budget or dense NxM tensor. The BVH result uses FP32 distance arithmetic.
- Zero-distance sqrt loss has explicitly finite zero coordinate derivative. First-order coordinate gradients only; no double-backward promise.
- Deterministic minimum-original-ID tie rule; arbitrary ties/FP32 rounding may still differ from another implementation.
- Default auto dispatch uses point-count pair budget, not labels. Explicit modes remain available and are benchmarked independently.

## Controlled benchmark

Two independent uniform FP32 3D clouds, same seed/range/nested prefixes as preceding scaling study. Synthetic volume, not LiDAR geometry.
Full-shape forward/backward warmup then3 measured repetitions. Synchronized wall time, includes allocation, loss and prediction-only backward; excludes model/I/O/DDP.
Every BVH/KD call builds fresh trees. Backend order rotates by size. This measures no-grad index selection plus differentiable loss, not Teacher optimizer throughput.
The tiled improvement and the additional BVH algorithmic gain are reported separately.

| N=M | Original CUDA(ms) | Scene tiled(ms) | Scene BVH(ms) | KeOps(ms) | Prior KD-tree(ms) |
|---:|---:|---:|---:|---:|---:|
|1024|1.642|2.289|2.766|5.394|10.933|
|2048|1.556|3.237|4.130|5.470|6.947|
|4096|1.482|2.027|3.076|4.383|6.623|
|8192|1.922|2.317|3.227|3.974|7.936|
|16384|5.093|2.956|3.791|7.902|10.532|
|32768|17.075|5.215|4.207|15.848|12.583|
|65536|54.822|14.254|5.548|46.220|16.105|
|131072|110.295|38.284|7.336|134.601|23.218|
|262144|646.569|141.325|12.846|499.640|49.214|
|524288|1750.865|545.386|22.679|1980.775|64.444|
|1048576|10318.401|2229.286|52.774|7898.867|136.295|

## Extra peak PyTorch allocated memory

Includes CD forward and backward workspace, excludes shared input baseline and network. Third-party raw CUDA allocations may not appear in Torch counters.
Full maximum-size backing point tensors are in the common input baseline for all sizes. Reserved cache is separately recorded in CSV.

| N=M | Original CUDA(MiB) | Scene tiled(MiB) | Scene BVH(MiB) | KeOps(MiB) | Prior KD-tree(MiB) |
|---:|---:|---:|---:|---:|---:|
|1024|0.048|0.063|0.066|0.080|0.106|
|2048|0.095|0.126|0.133|0.158|0.213|
|4096|0.188|0.251|0.266|0.381|0.426|
|8192|0.376|0.501|0.558|0.757|0.980|
|16384|0.751|1.001|1.107|1.509|1.951|
|32768|1.501|2.001|2.205|3.013|3.892|
|65536|3.001|4.001|4.401|6.021|7.776|
|131072|6.001|8.001|8.792|12.039|15.542|
|262144|12.001|16.001|18.000|26.072|33.076|
|524288|24.001|32.001|35.142|48.139|62.142|
|1048576|48.001|64.001|70.276|96.272|124.276|

At1048576x1048576, pure tiled optimization is4.63x versus original; BVH adds spatial pruning to reach195.52x in this run.
This is not a universal speedup: original timing fluctuated strongly and all min/max repetitions remain in CSV/JSON.
Max scalar CD difference versus CPU exact KD-tree across55 cases:1.7679764e-07m. All165 measured coordinate gradients finite; no OOM.
Scalar agreement and finite gradients do not establish gradient identity or convergence equivalence. Real Teacher endpoint parity is reported separately.
Small inputs may favor original CUDA. Pathological spatial distributions can weaken BVH pruning; nearest-neighbor gradients are nonunique at ties.
The custom CUDA backward accumulates atomically and is not guaranteed bitwise deterministic.

Artifacts: `/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/chamfer_scene_scaling_1024_to_1048576_gpu1` contains complete raw timing/memory runs,CSV,original hashes,and source snapshot.
No formal training pipeline was switched automatically.
