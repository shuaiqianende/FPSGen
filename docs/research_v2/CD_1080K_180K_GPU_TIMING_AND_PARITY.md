# CD1080000 target versus180000 prediction: timing and equality audit

PhysicalGPU1,FP32 inputs;18w prediction points and108w raw GT points. Real08/003900 coordinates.
To meet exact requested sizes, fixed seed20261019 selects180000 of504083 predicted fine centers and1080000 of1144476 valid raw-map points, WITHOUT replacement.
This is a size benchmark, not a replacement evaluation dataset. Full official CD still uses ALL valid original raw-map points.
No model forward/backward, optimizer, DDP or I/O is included in the CD timings. Source preparation and host transfers are excluded.
Full-size warmup precedes3 timed runs. New tree timing includes TWO fresh GPU trees, both nearest queries and differentiable coordinate norms.

| Backend | CD forward(ms) | CD backward(ms), prediction coordinates | Median forward+backward(ms) |
|---|---:|---:|---:|
|keops|1410.916|2.699|1413.929|
|chamfer_cuda|1829.542|1.479|1831.168|
|gpu_kdtree|88.119|2.581|90.700|

The total median is computed from per-run totals, not the sum of independent medians.
New total0.090700s versus KeOps1.413929s (~15.59x) and native CUDA1.831168s (~20.19x).
CPU exact tree build+query2.475068s is a single control timing, not a repeated median.
Pure evaluation FP64 tree CD including both builds0.096439s. Peak allocated0.141GiB.

## Numerical checks

All three differentiable FP32 losses:0.20444166660308838.
CPU FP64 reference CD:0.20444166331372704; GPU tree FP64 evaluation:0.20444166331372710.
Absolute evaluation difference5.55e-17m.1260000 individual nearest distances checked in both directions.
GPU tree maximum per-query distance error0.0m in this run. Existing FP32 native CUDA max error6.891e-7m.

## Exact tie and gradients

The initial STRICT gradient comparison against CPU failed: relativeL2=1.891e-4 exceeded the registered1e-4 threshold.
The original JSON remains `GRADIENT_REVIEW_REQUIRED`; the threshold was not changed and the failure is not relabeled.
A separate audit found one GT query(index283440) exactly equidistant from two prediction points. Both squared distances are0.0058702772783476576.
CPU selected prediction index6371; new GPU tree selected6368. Either is a true nearest neighbor. All actual distances are equal.
The resulting scalar loss agrees, but the piecewise coordinate gradient at this nondifferentiable tie is not unique.
Two prediction points have differing CPU versus GPU gradients, maximum absolute difference3.934e-7.
New GPU tree and ORIGINAL KeOps selected identical nearest indices for ALL1260000 queries.
Their recomputed-coordinate losses and gradients are exactly equal on this input (max gradient difference0).
Native CUDA also chose the same indices; its norm-recomputed gradient agrees, while native squared-distance/sqrt arithmetic may introduce ordinary rounding differences.
The independent followup status is `PASS_EXACT_DISTANCE_WITH_TIE_SUBGRADIENT`.
These statements describe tested inputs, not universal bitwise equivalence or unique derivatives at every possible tie.

## Evidence

`stage2_morph_fm100k_audit/cd_1080000_vs_180000_gpu1.json` contains all warmup/runs and the original strict gradient result.
`stage2_morph_fm100k_audit/cd_1080000_vs_180000_tie_audit_gpu1.json` contains exact tie coordinates, indices, squared distances and cross-GPU gradient comparisons.
Scripts: `benchmark_cd_1080k_180k.py` and `audit_cd_1080k_ties.py` under `scripts/research_v2/hier_occ_inference/`.
No formal Teacher/Stage2 training backend, weight, optimizer or checkpoint was changed. GPU2/3 jobs were not interrupted.
