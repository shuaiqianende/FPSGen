# GPU KD-tree tradeoffs: measured memory and counterexamples

PhysicalGPU1 only. No formal training loss/checkpoint was changed. Real memory benchmark uses the exact same fixed1080000-target/180000-prediction points as the preceding size benchmark.

## CD workspace memory

Each backend receives identical FP32 input tensors; input copies and common reference tensors form the live baseline.
Each trial clears inactive reserved cache, resets peak counters, then runs differentiable CD construction and prediction-coordinate backward.
Values are median additional PyTorch peak allocated bytes over baseline, not whole-process GPU memory.
Third-party raw CUDA allocations may be outside Torch counters. CUDA reserved cache is reported separately in JSON and is not live allocation.

| Backend | Extra peak allocated(MiB) | Total reserved peak(MiB),including baseline |
|---|---:|---:|
|chamfer_cuda|28.842|80.000|
|keops|75.125|142.000|
|gpu_kdtree|111.521|172.000|

New GPU KD-tree adds approximately82.68MiB versus native Chamfer CUDA and36.40MiB versus KeOps in this workload.
The tree needs reordered coordinates, original IDs, temporary sort keys/permutations and node bounding boxes. Added space grows with point count.
The previously measured16.50GiB is total Teacher profiling peak across backends, not the memory attributable to the KD-tree alone.

## Not always faster

Synthetic stress tests below are evaluation-only and deliberately small; native CUDA uses FP32 distance arithmetic, tree uses FP64.
Build/query/mean and synchronization included. Two warmups followed by5 repeated runs. These do not establish a universal crossover point.

| Test | Points P/GT | Native CUDA median(ms) | GPU tree median(ms) | Tree/native time ratio |
|---|---:|---:|---:|---:|
|tiny_random|64/64|0.2164|1.1311|5.23x|
|small_random|1024/8192|0.6105|12.3881|20.29x|
|all_identical_points|1024/8192|0.8751|28.0706|32.08x|

Small clouds do not amortize multiple GPU sort passes, kernel launches and Python orchestration.
The all-identical case gives every node the same zero-distance lower bound. With exact tie handling, the current query traverses all relevant nodes;
pruning fails and work approaches the all-pairs bound, with extra traversal overhead. Random small inputs are already slower due to build/launch overhead.
GPU thread divergence and cache-unfriendly traversal vary with point geometry. Million-point3D LiDAR speedup is not a guarantee for arbitrary clouds or dimensions.

## Dynamic training and derivatives

Predicted coordinates change each optimizer update, so their tree cannot be reused without an explicit validity mechanism.
Frozen GT indices might be cached, but that has not been added to production. Timings include fresh builds for both sides.
Nearest-neighbor assignment changes and exact ties are nondifferentiable for ANY nearest-neighbor CD.
Different libraries/tie/rounding rules can therefore agree on scalar CD yet produce different coordinate subgradients.
The1260000-input Teacher profile still has PARITY_REVIEW_REQUIRED: GPU-tree versus KeOps gradient relativeL2=4.539e-4 exceeds registered1e-4 tolerance (max absolute4.501e-7).
The cause for that particular endpoint has NOT been proven; do not automatically attribute it to exact ties identified in another experiment.
No automatic replacement of formal Teacher training backend is justified by the speed result alone.

## Implementation maturity and scope

Current implementation is3D CUDA FP32 coordinates, FP64 exact-distance query arithmetic, bounded validated tree depth, per-scene variable-length orchestration.
It is a custom prototype, not a drop-in general-purpose high-dimensional or arbitrary-backend KD-tree package.
Additional large-scene, extreme-input, gradient and training-convergence checks are needed before calling it a fully validated training backend.
No dense pairwise matrix is materialized and nearest search is not approximate. Speed comes from eliminating impossible spatial candidates, with real memory and traversal tradeoffs.

Evidence JSONs are copied into `stage2_morph_fm100k_audit/`:
`cd_1080000_vs_180000_backend_memory_gpu1.json`, `gpu_kdtree_small_and_degenerate_limits_gpu1.json`.
