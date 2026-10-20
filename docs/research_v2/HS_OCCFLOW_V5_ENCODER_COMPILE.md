# HS-OccFlow V5 Encoder Compilation

## Setup

Tested the four dense Stage1 Encoder components individually on GPU2, with the frozen Stage1/Stage2 checkpoints and the production V4 compiled-Flow + PTv3 AMP/multilevel-cache path. Encoder-only smoke used frame 100. The formal candidate benchmark used five real-frame warmups and the standard 20 timed frames. No `suppress_errors` fallback was enabled.

| Target | Lazy first-call / compile cost | Encoder smoke mean | Pyramid parity |
|---|---:|---:|---|
| `voxel_backbone` | 4.96 s | 203.75 ms | exact at all three outputs |
| `voxel_neck` | 0.22 s | 204.75 ms | exact |
| `local_backbone` | 6.00 s | 202.14 ms | exact |
| `classification_neck` | 19.57 s | 194.90 ms | first output rel-L2 `1.64e-4`; other two exact |

The single-frame timings were screening measurements, not sufficient evidence for accepting the candidate. The 20-frame `classification_neck` benchmark is at `experiments/hs_inference_opt_v5/encoder_compile/classification_neck_gpu2_round1_5x20/`.

## Formal 5+20 result

Compared against a same-GPU V4 compiled-Flow baseline, both using the unchanged Stage2 AMP/cache configuration:

| Mean | V4 baseline | Classification Neck compiled | Difference |
|---|---:|---:|---:|
| Stage1 Encoder CUDA Event | 203.739 ms | 194.495 ms | −9.244 ms |
| Stage1 Flow ×2 | 60.623 ms | 60.964 ms | +0.341 ms |
| Stage2 PTv3 ×4 | 143.205 ms | 144.855 ms | +1.650 ms |
| Full generation wall | 431.280 ms | 428.336 ms | −2.944 ms |

Paired per-frame full-generation difference had SD 7.77 ms and a 95% confidence interval half-width of 3.41 ms; the interval includes zero. Thus the Encoder saving is clear in this run, but a full-pipeline speedup is not established. Peak allocated memory stayed 3.334 GiB; reserved memory was about 3.978 GiB. Encoder compile added about 14.55 seconds before steady-state execution.

Compilation had numerical effects: Stage1 scores and candidate IDs differed on 20/20 frames; candidate count changed by a mean +3.1, and the mean fine-occupancy symmetric difference was 2,754 voxels. These are small relative changes but disqualify this as an exact engineering-preserving variant.

## 41-frame quality regression

Using the frozen Seq08/skip100 GT and shared metrics, the candidate versus the V4 compiled-Flow reference produced:

| Metric | Baseline | Compiled neck | Difference |
|---|---:|---:|---:|
| CD (m) | 0.202787 | 0.202845 | +0.000058 (+0.029%) |
| IoU@0.5m | 0.506824 | 0.506816 | −0.000009 |
| IoU@0.2m | 0.395352 | 0.395344 | −0.000008 |
| IoU@0.1m | 0.164191 | 0.164188 | −0.000003 |
| JSD-BEV | 0.307588 | 0.307625 | +0.000037 |
| JSD-3D | 0.416162 | 0.416168 | +0.000007 |

The quality differences are small and within the requested screening tolerances. This does not establish that the compile variant is a worthwhile default: full-generation speed did not show a statistically clear gain, and compile cold-start is material. We therefore keep it as an optional numerical variant, not the recommended path. The V4 Baseline remains the default.

Raw metric files: `experiments/hs_inference_opt_v5/quality_41/classification_neck_metrics.json` and `.csv`. Raw prediction/timing artifacts remain ignored local experiment outputs.

## Other compile targets

The single-frame target smokes show graph breaks/recompilations in `voxel_backbone` (6 unique graphs, 30 graph breaks) and `local_backbone` (9 unique graphs, 42 graph-break events). `voxel_neck` was exact but had no speed improvement. `classification_neck` compiled most of its graph but still changed the high-resolution feature slightly. No other target received a full 5+20 run because no evidence justified broadening the expensive benchmark.
