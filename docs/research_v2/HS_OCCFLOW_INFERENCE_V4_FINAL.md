# HS-OccFlow Inference Optimization V4 Results

## Frozen inference contract

- V4 branch: `research/hs-occflow-stage1-speed-v4`, based on V3 `f51a74096ce9bd4575164fce04e8f54c6fa540c3`.
- Stage1 EMA checkpoint SHA256: `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`.
- Stage2 Online checkpoint SHA256: `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`.
- Stage1=2 NFE, Stage2=4 NFE, threshold 0.5, batch 1, original Stage1 autocast, V3 GPU candidate/observed-mask/decode and Stage2 static cache + PTv3 multilevel cache/AMP FP16/batch-1 fastpath.
- Registered performance protocol: 5 warmups `[0,800,1600,2400,3200]`, 20 timed frames `[100,300,...,3900]`. All newly compared variants ran on the same physical RTX 3090 GPU2. Generation timings exclude disk I/O and quality metrics; `--skip-jitter` was used to isolate model generation.
- GPU3 was not used in this phase because it was actively occupied (about 7.5 GiB, 91–98% utilization). No running process was stopped.

## Main measured result

| Variant | Encoder ms | Flow ×2 ms | Stage2 ×4 ms | Network CUDA Events ms | Generation wall ms | Peak allocated GiB | Jitter |
|---|---:|---:|---:|---:|---:|---:|---|
| V3 eager, 3 runs | 203.71 | 107.99 | 143.13 | 454.83 | 478.12 | 3.011 | Not measured in these runs |
| Stage1 Flow compiled, 3 runs | 204.57 | 61.01 | 144.51 | 410.08 | **435.78** | 3.334 | Not measured in these runs |

The compiled Flow lowers generation wall time by `42.34 ms` or `8.86%` over three paired 5+20 rounds. Stage1 Flow time decreases by about `46.98 ms` (`43.5%`). The gain is repeatable across all three run means; encoder and Stage2 times are effectively unchanged. Peak allocated memory increases by about `0.323 GiB`. The compile-first forward takes about `10.13 s` on warmup and is not included in timed latency.

This reaches the requested ~400–500 ms generation range without changing NFE, checkpoints or network capacity. It is not an exact-parity transformation: compiled inference is a numerical variant and should remain opt-in.

## Optimization outcomes

| Optimization | Outcome |
|---|---|
| Stage1 static condition projection cache | Implemented and CPU tested. One GPU0 round measured only ~1.22 ms net Flow and ~2.43 ms end-to-end savings. Repeated full-run exact Occupancy parity was not established on GPU2; keep opt-in. |
| Encoder channels-last 3D | Rejected; Encoder +90.7 ms and full generation +91.1 ms on the screening run. |
| Flow channels-last 3D | Small/noisy ~4.1 ms full-path improvement; changed output hashes. 41-frame quality was essentially unchanged, but insufficient performance gain for default use. |
| cuDNN benchmark | Rejected; full generation 488.19 ms vs 478.07 ms baseline, Stage1 score/candidate hashes changed. |
| Stage1 TF32 | Rejected; full generation 482.80 ms vs 478.07 ms baseline, score/candidate hashes changed. |
| Stage1 Flow `torch.compile` | Best candidate; 3-run mean 435.78 ms. Numerical variation; 41-frame macro quality passed the initial engineering thresholds. |
| CUDA Graph | Not tested. Static Flow shapes make it plausible, but ordinary forward has a tensor-to-Python boolean check; compile already provides substantial measured savings. |
| Custom LiDAR tokenizer kernel | Not pursued; `unique` + `scatter_max` profiling totaled under 0.7 ms, not a meaningful hotspot. |

## 41-frame quality regression

The V3 baseline and compiled Flow predictions were evaluated on the same Seq08 skip100 frames, with the existing raw-map GT, fixed ROI, exact GPU KD-tree Chamfer, corrected physical-grid IoU and shared JSD. Results are macro averages:

| Metric | V3 Eager | Compiled Flow | Change |
|---|---:|---:|---:|
| CD (m) ↓ | 0.202997 | 0.202787 | −0.000210 |
| IoU@0.5 ↑ | 0.506869 | 0.506824 | −0.00445 percentage points |
| IoU@0.2 ↑ | 0.395382 | 0.395352 | −0.00305 percentage points |
| IoU@0.1 ↑ | 0.164187 | 0.164191 | +0.00045 percentage points |
| JSD-BEV ↓ | 0.307663 | 0.307588 | −0.000076 |
| JSD-3D ↓ | 0.416174 | 0.416162 | −0.000013 |

These changes satisfy the supplied initial screening tolerances. Compiled scores and candidates differ for all 41 frames; mean Candidate count changes by +37.7, and Fine Occupancy symmetric difference averages 5,797 cells (maximum 11,019). No large per-frame metric regression was observed, but the output is not bitwise identical.

## Artifacts and reproducibility

- Stage1 profile: `docs/research_v2/HS_OCCFLOW_STAGE1_PROFILE_V4.md`
- Flow condition cache: `docs/research_v2/HS_OCCFLOW_STAGE1_FLOW_CACHE_V4.md`
- Encoder layouts/cuDNN/TF32: `docs/research_v2/HS_OCCFLOW_ENCODER_LAYOUT_V4.md`
- Compile and graph audit: `docs/research_v2/HS_OCCFLOW_STAGE1_COMPILE_GRAPH_V4.md`
- 41-frame metrics JSON/CSV: `experiments/hs_inference_opt_v4/quality/stage1_flow_compile_gpu2_41/metrics.json` and `.csv`
- 41-frame predictions and timing metadata: `experiments/hs_inference_opt_v4/quality/stage1_flow_compile_gpu2_41/`
- Repeated 5+20 timing rows and summaries: the corresponding `experiments/hs_inference_opt_v4/{baseline,compile}/` paths listed in the compile report.
- V4 CPU tests: `pytest -q tests/test_stage1_layout_v4.py tests/test_stage1_flow_condition_cache_v4.py tests/test_stage1_occ_runner.py` (28 passed). GPU tests are reported separately; CPU tests are not presented as GPU validation.

Jitter was intentionally not rerun during network optimization, so the full 1.08M-point delivery latency for the compiled model is **NOT TESTED**. The previous jitter measurement cannot simply be added to a different run and presented as a new end-to-end measurement.

## Recommendation

Keep V3 eager behavior as the default for exact historical reproducibility. Offer Stage1 Flow compilation as an explicit opt-in numerical-performance mode after a release decision: it saves ~42 ms/frame with a small increase in memory and very small aggregate 41-frame metric changes. Do not claim exact parity, and do not enable cuDNN benchmark, TF32 or channels-last Encoder. Keep the condition cache opt-in because its net gain is only a few milliseconds and repeated Occupancy parity was inconclusive.
