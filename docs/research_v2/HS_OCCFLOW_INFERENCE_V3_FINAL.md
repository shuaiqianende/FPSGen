# HS-OccFlow Inference Optimization V3 Results

## Frozen protocol

- Branch base: `cfa13d0433d638789fcd3ed0a9e0a5ea8cbe2dea`; V3 work is isolated on `research/hs-occflow-ptv3-performance-v3`.
- GPU: NVIDIA RTX 3090, physical GPU3; PyTorch 2.0.1+cu117, CUDA runtime 11.7.
- Stage1: EMA checkpoint `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`, 2 NFE.
- Stage2: Online checkpoint `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`, 25,000 updates / 100,000 frame exposures, 4 NFE.
- Batch size 1, fixed Stage1 threshold 0.5, unchanged source noise and Euler updates.
- Three paired latency rounds used the same five warmup frames `[0,800,1600,2400,3200]` and twenty measured frames `[100,300,...,3900]`. Baseline and combined variants alternated on the same physical GPU. Jitter was omitted in these repeated network latency rounds and measured separately in one complete-output round.

## Three-round latency comparison

Values below are the mean of three run-level means. Each run contains 20 timed frames. The combined configuration is `multilevel` PTv3 topology cache + AMP FP16 + validated batch-1 fastpath. Baseline keeps original PTv3 precision and no topology cache.

| Variant | Stage1 encoder (ms) | Stage1 flow ×2 (ms) | Stage2 PTv3 ×4 (ms) | Network CUDA events (ms) | Full generation wall (ms) | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| V2 eager baseline | 204.16 | 107.95 | 304.47 | 616.58 | 632.79 | 3.01 |
| AMP + multilevel cache + batch1 fastpath | 203.84 | 107.68 | 143.23 | 454.75 | 480.93 | 3.01 |

Across the three run means, full generation fell by **151.86 ms / 24.0%** and
Stage2 four-step time fell by **161.24 ms / 53.0%**. Run-level generation
standard deviation was 6.84 ms for baseline and 5.06 ms for the optimized
variant. Peak allocated memory did not change. Per-run and pooled per-frame
statistics (including median, P90 and P95) are in
`experiments/hs_inference_opt_v3/repeats/three_round_summary.json`.

The first complete-output run, which also emitted 1,080,000 Native Fine Jitter
points, measured 486.62 ms generation + 452.03 ms jitter = 938.64 ms total,
with 3.01 GiB peak allocated / 3.84 GiB reserved. The earlier same-GPU
baseline output run measured 633.66 ms generation + 405.95 ms jitter =
1,039.61 ms. Jitter has high per-frame variability, so the generation speedup
is the reliable network result; the full 1.08M output path improved by about
9.7% in this one paired-protocol run.

The 0.4–0.5 s generation target was reached on average; P95 generation in the
first combined run was 523.43 ms. It is not a per-frame upper bound.

## What each optimization contributed

| Experiment | Result |
|---|---|
| Top-level serialization cache | On GPU3, Stage2 fell from 301.93 to 263.54 ms and generation from 633.66 to 604.70 ms. Fine Occupancy matched baseline exactly on 20/20 frames. |
| Attention padding + pooling topology maps | Multilevel cache brought Stage2 to 258.47 ms and generation to 599.07 ms, also exact on 20/20. Its ~5 ms extra gain over serialization alone is small and should not be treated as a robust standalone win. |
| AMP FP16 alone | On GPU2, paired same-GPU baseline/AMP generation means were 638.87/524.22 ms; Stage2 was 307.71/193.60 ms. It ran without dtype errors. 41-frame quality passed. |
| Selective FP16 | On GPU3, generation was 525.93 ms and Stage2 195.42 ms. Quality passed, but speed was effectively tied with AMP; no clear reason to prefer its added norm policy. |
| Batch-1 fastpath alone | Exact Occupancy parity, but generation improved only 0.6% in its standalone GPU3 test. It is retained in the best combination because it is safe and may help in combination; no independent benefit is claimed. |
| SpConv indice cache | Not implemented. SpConv compute is a significant share of Stage2, but safe reuse of its mutable indice metadata was not established. |
| `torch.compile` Stage2 MLP | Ran, but was slower: 20-frame generation 750.18 ms and Stage2 321.86 ms versus GPU3 eager 633.66 and 301.93 ms. Four Dynamo graphs were created; the first call took 8.50 s. Rejected. |
| `torch.compile` Stage1 Flow | Smoke failed in PyTorch 2.0.1 Inductor at Stage1 timestep-frequency indexing with `Inference tensors do not track version counter`. No eager fallback was enabled. |
| CUDA Graph | Not tested; variable Parent token counts and dynamic PTv3 sparse topology make a semantics-preserving capture path nontrivial. |

The three-scene PTv3 profile measured 39.9–46.4 ms of serialization CUDA
events over four repeated forwards. SpConv module events were 51–86 ms,
attention 35–68 ms, pooling 8–13 ms and unpooling 6–11 ms as token count grew
from about 78K to 170K. The exact profile is in
`experiments/hs_inference_opt_v3/profile/baseline_ptv3_3scenes.json`.

The benchmark also reports about 50.2 ms of candidate-selection host submit
time but only about 0.7 ms between its CUDA events. This gap points to
CPU-side dynamic-shape/synchronization work in candidate creation; the current
measurements do not isolate a single offending operator, so it should not be
misreported as 49 ms of GPU compute. The batch-1 PTv3 fastpath had negligible
standalone benefit and did not resolve this candidate-generation overhead.

## Precision and quality

The source/runtime dtype audit is documented in
`HS_OCCFLOW_ACTUAL_PRECISION_AUDIT.md`. Stage1 encoder and flow already use
mixed CUDA autocast; output pyramid levels were `[FP32, FP16, FP16]`. Original
Stage2 PTv3 runs its embedding, sparse convolutions, attention projections and
normalization in FP32, while its FlashAttention kernel explicitly casts QKV to
FP16 internally. AMP keeps FP32 weights and Euler state while enabling
autocast only around PTv3.

All three quality comparisons use the same 41 Seq08/skip100 frames, identical
raw-map GT, exact CUDA KD-tree CD, corrected physical-grid IoU and shared JSD:

| Variant vs original | CD Δ (m) | IoU@0.5 Δ | IoU@0.2 Δ | IoU@0.1 Δ | JSD-BEV Δ | JSD-3D Δ |
|---|---:|---:|---:|---:|---:|---:|
| AMP FP16 | +0.00000084 | +0.00121 pp | +0.00033 pp | +0.00010 pp | +0.00000007 | -0.00000395 |
| Selective FP16 | -0.00000092 | +0.00129 pp | +0.00029 pp | +0.00012 pp | -0.00000008 | -0.00000276 |
| Combined best | -0.00000128 | +0.00114 pp | +0.00031 pp | +0.00014 pp | -0.00000351 | -0.00000367 |

The combined best variant passed the requested quality screen and showed no
meaningful metric degradation. Its exact comparison is
`experiments/hs_inference_opt_v3/quality/combined_amp_multilevel_seq08_skip100_metrics.json`;
AMP and selective results have matching JSON/CSV records alongside it.

## Recommendation and limits

Use `amp_fp16` plus `multilevel` topology cache as the tested fast inference
configuration when the 41-frame quality tradeoff is acceptable. Keep all V3
switches opt-in in code so the default remains the original V2 path. Do not
enable `torch.compile` based on these results. The cache saves repeated
geometry work but does not reduce dynamic Attention, SpConv or velocity
computation. No NFE, checkpoint, candidate policy or model capacity changed.

The main benchmark used one 20-frame sample set over three alternating rounds;
the 41-frame metric test is a fixed Seq08 development-style regression, not a
multi-sequence generalization claim. No full 408-frame skip10 evaluation or
training was run. No other GPU jobs were interrupted.
