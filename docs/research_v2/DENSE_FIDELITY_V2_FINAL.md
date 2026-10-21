# HS-OccFlow Dense Fidelity V2 — Results

**Status: 20K training, 900-frame quality evaluation, PyTorch timing, and E3 TensorRT FP16 validation are complete.** No 100K training or Stage2 retraining has started.

## Candidate quality at 20K

All candidates used fresh Stage1 initialization and the registered 20,000 successful frame exposures / 5,000 updates. Quality values are macro averages over the same 900-frame development Manifest, with EMA, NFE=2, strict score threshold `> 0.5`, and fixed candidate budgets selected before comparing variants.

| Variant | Threshold IoU | Threshold F1 | Fine-GT ceiling | Fixed-K IoU, K=78,694 | K=84,963 | K=94,087 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| R0 Original | 0.400207 | 0.567894 | 0.573662 | 0.360863 | 0.371353 | 0.382981 | Reference |
| E1 Bottleneck-96 | 0.24872 | 0.39418 | 0.46979 | Not evaluated | Not evaluated | Not evaluated | Reject: major quality loss |
| E2 Asymmetric3D rank-64 | 0.390837 | 0.557872 | 0.556141 | 0.358537 | 0.368421 | 0.379314 | Reject: misses registered threshold-IoU and recall-ceiling gates |
| E3 Lightweight Skip | 0.395728 | 0.563331 | 0.571696 | 0.356640 | 0.367221 | 0.379015 | Passes registered minimum screening gates; retain as speed-quality candidate |

The preregistered screening floors were threshold IoU >= 0.395, fixed-K IoU at K=84,963 >= 0.365, and Fine-GT ceiling >= 0.563. E3 passes all three. Relative to R0, E3 is lower by 0.00448 threshold IoU and 0.00413 fixed-K IoU at K=84,963; its Fine-GT ceiling is lower by 0.00197. This is a measurable quality tradeoff, not parity with R0.

## Encoder speed

R0 and E3 trained checkpoints were timed on the same RTX 3090 GPU2, with the same 25-row development input Manifest (5 warmup, 20 timed). The Manifest SHA256 is `224c556d7a8da18e437e53e2e67aa08cf364c909f80fe068dbb813570c7955fe`.

| Variant / backend | Encoder CUDA event mean | Encoder wall mean incl. H2D | P95 wall | Peak allocated |
|---|---:|---:|---:|---:|
| R0 Original, PyTorch explicit FP16 | 179.78 ms | 180.14 ms | — | 1.940 GiB |
| E3 Lightweight Skip, PyTorch explicit FP16 | 167.07 ms | 167.45 ms | — | 1.939 GiB |
| R0 Original, TensorRT FP16 Local Backbone | 146.97 ms | 147.24 ms | 149.19 ms | 1.923 GiB |
| E3 Lightweight Skip, TensorRT FP16 Local Backbone | 136.81 ms | 137.09 ms | 138.19 ms | 1.923 GiB |

On PyTorch, E3 is 12.70 ms (7.58%) faster than R0. With the same TensorRT backend, E3 is 10.15 ms (6.90%) faster than R0. TensorRT lowered E3's full Encoder time by 30.35 ms (18.1%) relative to PyTorch explicit FP16. These are Encoder-only measurements; no full model generation speedup is claimed because the 20K Stage1 candidates were not adapted with Stage2.

TensorRT 10.13.3.9 built static FP16 engines for the Local Backbone. The E3 engine build took 67.34 s and measured 32.83 ms for the standalone backbone; R0 took 67.97 s and 36.99 ms. TensorRT Engine Inspector confirmed FP16 `CaskConvolution` tactics on the target 3D convolutions. The E3 outputs differed from PyTorch explicit FP16 by relative L2 0.00326 / 0.00295 / 0.00278 across its three output pyramid levels.

E3's full 900-frame TensorRT quality run and PyTorch explicit-FP16 control used the same development Manifest, same GT, NFE=2, score threshold and fixed candidate budgets. Macro threshold IoU was 0.39573624 for TRT and 0.39572834 for PyTorch; fixed-K IoU at K=84,963 was 0.36722667 versus 0.36722689. The threshold-IoU frame delta ranged from -0.0001221 to +0.0001234. Active-parent counts differed by a mean of 6.96 (range -71 to +67). The outputs are numerically variant, but aggregate geometry metrics are effectively unchanged. Exact per-frame parent-ID symmetric differences were not saved and are not claimed.

Detailed export, engine, and quality provenance is in [DENSE_FIDELITY_V2_TENSORRT.md](DENSE_FIDELITY_V2_TENSORRT.md). Raw machine-readable measurements remain under `experiments/dense_fidelity_v2/` and large checkpoint/engine files are excluded from Git.

## Research decision

E3 is the only new architecture that passes the registered 20K quality screen, and its encoder speedup is reproducible under both PyTorch FP16 and TensorRT FP16. It is a reasonable candidate for a controlled longer-training study, but current evidence does not establish that it is a better final model than R0: R0 retains higher Stage1 geometry quality, and Stage2 has not been retrained for E3. A next study should use a matched longer-training R0 control and E3, then adapt/evaluate Stage2 only after the encoder comparison remains favorable.

## Provenance

- E1, E2, E3 checkpoints and training records: `experiments/dense_fidelity_v2/training/`.
- Registered fixed-K results: `experiments/dense_fidelity_v2/fixed_k/paired_r0_e2_e3_900frames.json`.
- E3 TensorRT and PyTorch explicit-FP16 quality results, timing outputs, ONNX exports and engine inspector reports: `experiments/dense_fidelity_v2/tensorrt/` (large binaries remain local).
- R0 20K checkpoint was read from the separate OMD V1 worktree and was not copied into this branch.
- No checkpoint or historical V4/V5/V6/V7 experiment result was modified.
