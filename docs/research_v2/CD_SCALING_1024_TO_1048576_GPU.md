# CD scaling1024x1024 through1048576x1048576

COMPLETE:11 power-of-two sizes x3 backends,99 measured repetitions. PhysicalGPU1,RTX3090,B1.
Two independent uniform FP32 XYZ clouds over[-50,50]x[-50,50]x[-4,4.4],seed20261019, nested prefixes.
Synthetic volumetric clouds, not real LiDAR surfaces. Results are distribution/implementation/hardware specific.
Every backend receives exactly the same two clouds at a given size. No duplicate padding or approximate nearest search.
Full-size forward/backward warmup is excluded. Three synchronized timed repetitions per pair; rotate backend order across sizes.
Training loss is0.5*(mean Euclidean P->GT + GT->P). Only prediction coordinates require gradients.
Each GPU-tree CD includes fresh construction of BOTH trees. No tree cache is reused.
Network, optimizer, DDP, input generation/transfers and CPU validation are excluded.
Inactive Torch cache is cleared before measured trials: allocation work is included even though code/kernels have been warmed.
Reported total is median per-run forward+backward wall time; independent forward/backward medians need not sum exactly.

| Points per cloud | Chamfer CUDA total(ms) | KeOps total(ms) | GPU KD-tree total(ms) |
|---:|---:|---:|---:|
|1024|1.600|3.387|5.469|
|2048|1.627|3.425|6.363|
|4096|1.690|3.763|6.799|
|8192|2.002|4.459|8.136|
|16384|4.483|7.957|10.157|
|32768|9.848|16.347|12.897|
|65536|48.433|39.817|17.767|
|131072|161.829|136.774|36.845|
|262144|436.009|497.947|48.506|
|524288|1756.386|1966.155|63.810|
|1048576|10341.197|7797.147|134.692|

## Additional live tensor memory

Extra peak PyTorch allocated above input baseline, forward AND backward included. MiB=2^20bytes.
This excludes shared inputs and all Teacher/network memory; it is not total nvidia-smi memory.
Torch reserved memory includes cache and is reported separately in CSV. Raw third-party CUDA allocations may be outside Torch counters.
The full maximum-size backing tensors used to generate nested prefixes remain in the input baseline even for small N.

| Points per cloud | Chamfer CUDA extra(MiB) | KeOps extra(MiB) | GPU KD-tree extra(MiB) |
|---:|---:|---:|---:|
|1024|0.048|0.080|0.106|
|2048|0.095|0.158|0.213|
|4096|0.188|0.381|0.426|
|8192|0.376|0.757|0.980|
|16384|0.751|1.509|1.951|
|32768|1.501|3.013|3.892|
|65536|3.001|6.021|7.776|
|131072|6.001|12.039|15.542|
|262144|12.001|26.072|33.076|
|524288|24.001|48.139|62.142|
|1048576|48.001|96.272|124.276|

## Forward and backward separately

| Points per cloud | Backend | Forward(ms) | Backward(ms) |
|---:|---|---:|---:|
|1024|chamfer_cuda|0.468|1.152|
|1024|keops|1.831|1.556|
|1024|gpu_kdtree|4.036|1.443|
|2048|chamfer_cuda|0.463|1.164|
|2048|keops|1.932|1.491|
|2048|gpu_kdtree|4.786|1.430|
|4096|chamfer_cuda|0.533|1.157|
|4096|keops|2.209|1.558|
|4096|gpu_kdtree|5.293|1.502|
|8192|chamfer_cuda|0.808|1.195|
|8192|keops|2.770|1.689|
|8192|gpu_kdtree|6.580|1.580|
|16384|chamfer_cuda|3.242|1.232|
|16384|keops|5.950|2.021|
|16384|gpu_kdtree|8.602|1.536|
|32768|chamfer_cuda|8.220|1.624|
|32768|keops|14.687|1.711|
|32768|gpu_kdtree|11.316|1.581|
|65536|chamfer_cuda|46.892|1.541|
|65536|keops|38.213|1.597|
|65536|gpu_kdtree|16.616|1.151|
|131072|chamfer_cuda|160.612|1.225|
|131072|keops|134.617|2.206|
|131072|gpu_kdtree|34.957|1.794|
|262144|chamfer_cuda|434.694|1.281|
|262144|keops|495.888|2.022|
|262144|gpu_kdtree|46.746|1.760|
|524288|chamfer_cuda|1754.450|1.936|
|524288|keops|1964.427|1.570|
|524288|gpu_kdtree|62.537|1.236|
|1048576|chamfer_cuda|10338.007|3.190|
|1048576|keops|7795.157|2.466|
|1048576|gpu_kdtree|132.902|1.790|

## Correctness and limits

All33 scalar CD results agree with full-cloud CPU cKDTree(eps0) within1e-5m; maximum absolute error1.7679764e-07m. All99 measured prediction gradients finite. No OOM.
This scaling sweep validates loss values and finite gradients; it does not prove gradient identity across backends.
Previous large-context Teacher strict gradient comparison remains unpassed and is unaffected by this performance sweep.
Different nearest-neighbor tie rules and FP32/FP64 search rounding can produce different valid coordinate subgradients.
Only independent volumetric XYZ clouds are used here. Surface geometry, repeated points and strongly asymmetric N/M can change the crossover and worst-case behavior.
See GPU_KDTREE_TRADEOFFS_AND_MEMORY.md for measured small/duplicate-point counterexamples.

Artifacts: `/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/cd_scaling_1024_to_1048576_gpu1`; raw runs in results.json, full breakdown in summary.csv, log-scale plot in scaling.png.
No formal training sources, configurations, optimizers or checkpoints were changed. GPU2/3 training was not interrupted.
