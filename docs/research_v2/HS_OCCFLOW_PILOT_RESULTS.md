# Smoke and pilot status — no proposed-model quality claim

The new and legacy geometry/voxel suites pass64 tests;1 CUDA parity test is skipped.
Both real PTv3 variants construct on CPU; Stage2 construction forbids external
checkpoint loading in tests. Frozen Stage1 weights load strictly and stay frozen.
GPU forward/backward, FlashAttention runtime, AMP and DDP have **not** been tested
for HS-OccFlow because the allowed GPUs are occupied.

## CPU full-scene Local-FM mechanics smoke

This is an extra observed-mask-only ablation, width16, without PTv3 or LiDAR
pyramid. It uses8 real training frames and all occupied parents plus25% nearby
empty parents. Full scene sizes are50,452–159,628 tokens; no spatial truncation.
32 successful FP32 updates completed, gradients finite and nonzero. FM loss
1.999901 → 1.983721. Median CPU step was1.483s.
The EMA development2 results at1/2/4/8NFE have native macro IoU~0.24555 and recall
~0.4993, essentially a random-like initial endpoint. **This is not useful occupancy
generation or evidence that FM beats BCE.** It does not satisfy the proposed PTv3
overfit/pilot gate and none of its weights will initialize a formal model.

Continuous32 vs fresh20+resume12 are bit-identical for model, EMA, optimizer,
scheduler, scaler state, RNG, sampler cursor, loss sequence and frame order.
Model tensor SHA: `177008a6ec8185aa081175c9a845a4bb1156c0af927446137619dee8f119ac75`.
See [real resume audit](hs_occflow_reports/cpu_real_resume_audit.json).

Two engineering issues were found and corrected: disabled CPU autocast still
selected FP16 in Torch2.0; and sandbox IPC disallowed multi-worker tensor sharing.
The first failed step0 output is preserved; the corrected CPU run used a fresh
directory. I/O reruns used host IPC permissions and no GPU.

## Requested GPU experiments

| Experiment | Actual updates | Status |
|---|---:|---|
| Oracle PTv3+BCE8–16-frame overfit |0|blocked by GPU admission|
| Oracle PTv3+FM matched overfit |0|blocked by GPU admission|
| LiDAR-pyramid Local FM1K pilot |0|blocked by GPU admission|
| Matched BCE/FM1K, then5K pilots |0|not started|
| Formal experiment |0|not ready|

CPU smoke/temporary synthetic unit tests are not substituted for these experiments.
The gate remains `formal_ready=false`.
