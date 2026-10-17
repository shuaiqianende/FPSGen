# Stage2 384 resource and engineering audit

## CPU work actually executed

Full training-token audit: **17,870 real frames**, raw-map sparse labels,
94.68 seconds with four CPU workers. Development metadata contains 900 frames,
guard 360, and seq08 is excluded. The Oracle count includes all positive parents
and floor(0.25 × positives) neighboring empty parents; availability was proved
per frame, not assumed.

| Quantile | GT parents | Oracle parents including negatives |
|---|---:|---:|
| P50 | 138,000 | 172,499.5 |
| P90 | 207,215.2 | 259,018.5 |
| P99 | 259,269.37 | 324,086.04 |
| Maximum | 319,738 | **399,672** |

Maximum scene: sequence02 / frame000925, train index6147,
79,934 negative parents, 1,369,124 occupied fine voxels.
Other benchmark representatives: 02/004289 (P50), 06/000444 (P90),
02/003360 (P99). Predicted token quantiles are **not measured**; Oracle
statistics cannot replace a frozen-Stage1 prediction audit.

The isolated environment contains Python3.9, Torch2.0.1+cu117, MMCV1.7.2,
SpConv2.3.6, FlashAttention2.5.9.post1 and torch-scatter2.1.2+pt20cu117.
The alternate PointFlow environment lacks MMCV, so it is not this task's runner.
No system/main environment was modified and no packages were installed.

## Actual GPU validation: PASS on GPU2 andGPU3

Stage1 finished its100K successful frame budget at34,275 updates. GPUs were
verified idle before starting Stage2. The final frozen EMA checkpoint SHA256 is
b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2.
The earlier approval-service credit failure no longer blocks current execution.

Each GPU executed16 full-scene forward/backward/AdamW steps: BCE/FM ×
P50/P90/P99/maximum scenes × Oracle/Predicted support. No token reduction.
Actual FlashAttention and SpConv calls, finite gradients and single Stage1
encoding were verified. All Stage1 parameters remained frozen and unchanged.

| GPU | Objective | Maximum tokens | Peak allocated GiB | Peak reserved GiB | Maximum-scene training step s |
|---|---|---:|---:|---:|---:|
|2|BCE|399,672|6.043|8.037|0.692|
|3|BCE|399,672|6.043|8.037|0.706|
|2|FM|399,672|7.619|8.975|1.837|
|3|FM|399,672|7.619|8.975|1.837|

Times in this table exclude Stage1/candidate/feature assembly. The maximum FM
Oracle scene adds about1.11s assembly in the single-GPU benchmark. Training
DDP steady-state maximum-scene updates are about2.7–3.2s including assembly;
cold first updates include CUDA/DDP setup and are substantially slower.

Parent permutation error is exactly0 for both objectives on172,498 tokens.
Checkpoint arithmetic parity uses a separately labelled2048-token diagnostic;
actual gradient errors were2.9e-11 BCE and≤1.5e-10 FM. Resource rows above
retain all real tokens. Registered tolerances were unchanged (atol1e-5,rtol1e-3;
permutation max error<1e-4). First two failed attempts were preserved; upstream
Pooling shuffled orders despite the top-level flag, corrected in Stage2 only.

Real DDP continuous21 versus20+restore1 passed model, EMA, AdamW, scheduler,
scaler, two-rank RNG, sampler and synchronization checks. Stage2 loss atupdate21
was1.89479398727417 in both executions. The queue continues to32 successful
DDP updates before overfit. Full GPU reports and resume comparison are copied
under stage2_384_audit; large checkpoints remain ignored.

Predicted supports on the four engineering scenes contained136,996–172,048
parents at threshold0.5/NFE4. Parent recall was0.407–0.657; fine recall ceiling
was0.491–0.739. These training representatives are diagnostic, not a frozen
development calibration or full-training predicted maximum. Stage1 candidate
recall needs explicit development analysis before formal Predicted training.

SpConv executesFP32 internally; FlashAttention kernels useFP16. No system
environment changes, GPU0/1 usage, or interruption of other jobs occurred.
