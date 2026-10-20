# PTv3 Mixed Precision Results

The original PTv3 path explicitly casts input features to FP32 and disables
autocast around embedding, encoder and decoder. The FlashAttention branch casts
packed QKV to FP16 internally, then casts the result back. The frozen Stage1
encoder already runs under CUDA FP16 autocast; this was not a new optimization.

## AMP FP16

On RTX 3090 GPU2, the same registered 5 warmup + 20 measured frames completed
with `--ptv3-precision amp_fp16`. Stage2 four-step mean fell from 307.71 ms in
the same-GPU baseline to 193.60 ms (-37.1%). Full generation fell from 638.87
to 524.22 ms (-18.0%). The CUDA network event mean was 505.73 ms. Peak allocated
memory stayed at 3.01 GiB (reserved about 3.84 GiB). The 1.08M jitter postprocess
was 421.28 ms and is not network time.

The AMP path ran without dtype errors or non-finite output. It is a numerical
variant: on frame 100, 803,307 baseline occupied children versus 803,364 AMP
children, with symmetric difference 122. It is not marked exact. A paired
41-frame CD/IoU/JSD comparison is running; AMP is not recommended as the
default until those metrics finish.

## Selective FP16

`selective_fp16` retains half activations but performs custom Point Instance
Norm moment reductions in FP32, then casts normalized activations back before
affine application. A CPU unit test checks this behavior. GPU latency and
quality have now been measured on GPU3. On its 20 timed frames, Stage2 took
195.42 ms/frame, full generation 525.93 ms/frame, and generation plus native
jitter 973.02 ms/frame; peak allocation was 3.01 GiB. Same-GPU3 original
baseline was 633.66 ms generation and 301.93 ms Stage2, so the generation
reduction is 17.0%. Jitter is highly variable and excluded from that speedup.

41-frame evaluation completed with the same GT/metrics as the AMP run:

| Metric | Original | Selective FP16 | Delta |
|---|---:|---:|---:|
| CD (m) | 0.20299821 | 0.20299730 | -0.00000092 |
| IoU@0.5m | 0.50685739 | 0.50687025 | +0.00001286 |
| IoU@0.2m | 0.39537928 | 0.39538219 | +0.00000291 |
| IoU@0.1m | 0.16418529 | 0.16418644 | +0.00000116 |
| JSD-BEV | 0.30766700 | 0.30766691 | -0.00000008 |
| JSD-3D | 0.41617778 | 0.41617501 | -0.00000276 |

Selective FP16 also passes the configured 41-frame quality screen. Its mean
latency was effectively tied with AMP; there is no measured speed advantage
to justify the extra policy complexity yet. Full paired rows are in
`experiments/hs_inference_opt_v3/quality/selective_fp16_seq08_skip100_metrics.json`.

## 41-frame quality regression

All 41 Seq08/skip100 frames were evaluated against identical raw-map GT using
the same exact CUDA KD-tree CD, corrected physical-grid IoU and JSD functions.
The table reports macro values; delta is AMP minus original:

| Metric | Original | AMP FP16 | Delta |
|---|---:|---:|---:|
| CD (m) | 0.20299821 | 0.20299905 | +0.00000084 |
| IoU@0.5m | 0.50685739 | 0.50686947 | +0.00001208 |
| IoU@0.2m | 0.39537928 | 0.39538262 | +0.00000334 |
| IoU@0.1m | 0.16418529 | 0.16418631 | +0.00000102 |
| JSD-BEV | 0.30766700 | 0.30766707 | +0.00000007 |
| JSD-3D | 0.41617778 | 0.41617383 | -0.00000395 |

Occupied-cell count increased by 22.5 per frame on average. These differences
are negligible against the configured quality gates, so AMP FP16 passes this
41-frame quality screen. This is a quality result for the fixed Seq08/skip100
set, not a claim about all sequences. Full per-frame evidence is in
`experiments/hs_inference_opt_v3/quality/amp_fp16_seq08_skip100_metrics.json`
and its CSV sibling.

The earlier one-frame smoke also showed the same direction: CD +0.000028 m,
IoU@0.1 -0.0000021 and JSD-3D +0.0000106 for frame 0.
