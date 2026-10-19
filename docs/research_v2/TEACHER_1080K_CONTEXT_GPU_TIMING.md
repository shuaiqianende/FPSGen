# Teacher180K source +1080K GT context: full network profiling

Measured on physicalGPU1 only; same14,708,739-parameter Mapping Teacher checkpoint, FP16 AMP, FlashAttention and existing block activation checkpointing.
Input source180000 + GT context1080000 =1260000 points. Output remains180000 source-indexed endpoints.
GT context uses a fixed seeded subset WITHOUT replacement of1144476 valid raw-map points from08/003900.
Source anchors use the corresponding Poisson GT count distribution allocated to exactly180000 points. This is profiling, not new formal training or a quality comparison.

## Shape guard adaptation

The worktree legacy MappingTeacher rejects context lengths other than180000. The initial attempt stopped at this guard before network computation.
That evidence is retained as BLOCKED_BY_LEGACY_CONTEXT_SHAPE_GUARD. An independent profiling-only subclass copies the same forward feature/flag/center logic
while accepting1080000 context points. The PTv3 network, state_dict keys/shapes, parameter count and checkpoint weights are unchanged.
Formal source-hashed training code was not changed. No point count reduction or scene cropping was used to make the network fit.

## Complete forward/backward

Each backend has a separate full warmup and3 synchronized timed trials. Values below are independent medians.
Teacher backward includes activation-checkpoint recomputation. CD forward includes both searches/tree builds and loss construction.
CD backward differentiates prediction endpoint coordinates; Teacher backward consumes that gradient.
Timing excludes disk reads, model loading, first-call compilation, optimizer updates, DDP communication, checkpoint writes and finiteness scans.

| Backend | Teacher forward(s) | CD forward(s) | CD backward(s) | Teacher backward(s) | Total compute(s) |
|---|---:|---:|---:|---:|---:|
|keops|0.591014|1.429785|0.001775|1.445080|3.468761|
|chamfer_cuda|0.589403|1.852489|0.001284|1.446940|3.892058|
|gpu_kdtree|0.593895|0.068584|0.001699|1.449084|2.112246|

GPU-tree total compute is39.11% shorter versus KeOps (throughput factor1.642x).
Network-only forward about0.59s and backward about1.45s. Peak allocated16.501GiB; full source/context forward AND backward passed without OOM.
All timed parameter gradients were finite. First backward25.01s is preserved as warmup, excluded from the steady-state comparison.

## Numerical limits — strict gradient gate remains unpassed

The isolated SAME-endpoint FP32 scalar loss is0.06746306270360947 for all three backends.
Native CUDA versus KeOps endpoint gradient relativeL2=8.836e-8 and max absolute1.819e-12 (PASS).
GPU-tree versus KeOps endpoint gradient relativeL2=4.539e-4 and max absolute4.501e-7.
RelativeL2 exceeds the registered1e-4 tolerance: report remains PARITY_REVIEW_REQUIRED. Do not describe this large-context result as gradient-identical.
The nearest-index differences for this specific Teacher endpoint still require localization; no tie explanation is assumed without checking.
Independent full network forwards vary in loss by a few1e-7 even with seeds reset, so backend parity is evaluated on the same saved endpoint, not separate stochastic/nondeterministic forwards.
The previous voxel-center benchmark proved one exact tie on its own different point sets; that does not automatically prove this case has the same cause.
No production loss/backend replacement or training restart was performed.

## Artifacts

`stage2_morph_fm100k_audit/teacher180k_source_1080k_context_variable_adapter_gpu1.json` contains full warmup/runs, memory and unpassed strict gradient check.
`stage2_morph_fm100k_audit/teacher1080k_context_initial_guard.json` retains the original shape-guard rejection.
Script: `scripts/research_v2/hier_occ_inference/benchmark_teacher_1080k_context.py`.
