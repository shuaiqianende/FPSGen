# Stage-1 BEV Backbones: DiC-S-BEV and PixelU-S-BEV

Status: code preparation is complete.  GPU1 compute-only speed probes have been
run; no GPU correctness smoke, formal training, or BEV-quality evaluation has
been run.

## Fixed Stage-1 contract

All three options (`legacy`, `dic_s`, `pixelu_s`) use the existing direct
three-channel pixel-space BEV representation `[D,H,M]` at `3x256x256`.  They
share the same `DynamicKNNPillarEncoder` (`pcd_part -> [B,32,256,256]`), eight
uniformly sampled LiDAR/vehicle/road condition states, linear Flow Matching
path, velocity target, and channel-weighted `[1,2,1]` MSE.  No VAE, latent
representation, REPA, x-prediction, frequency loss, or altered condition
sampling has been introduced.

Inactive LiDAR, vehicle and road inputs are zeroed **before** the condition
adapter.  Both new adapters use only bias-free maps, so their outputs satisfy
`f_cond(0) == 0` exactly.  Learned scalar gates initialize at zero, so the
new spatial condition contribution is initially closed.

## DiC-S-BEV correspondence

Reference: `YuchuanTian/DiC`, commit
`bfa541b5e3a968919f3a399fc0e223e877439b3b`, `dic_models.py`, `DiC_S`.

| Official component | FPSGen implementation |
| --- | --- |
| `OverlapPatchEmbed` | Same 3x3, stride-1 entry map |
| `UNetBlock` | Same GroupNorm, GELU, adaptive scale/shift/gate, two 3x3 convs |
| `Downsample` / `Upsample` | Same 3x3 plus PixelUnshuffle / PixelShuffle |
| `TimestepEmbedder` / `FinalLayer` | Same topology; FPSGen maps `t` to `1000t` |
| DiC-S depth | `[6,6,5,6,6]` unchanged |
| DiC-S channel multiples | `[1,2,4,2,1]` unchanged |
| sparse skip schedule | `skip_stride=3` unchanged |
| ImageNet label embedding | Fixed zero; replaced by spatial FPSGen condition adapter |
| 4-channel sigma prediction | Necessary task change: 3-channel velocity output |

The DiC BEV condition adapter emits `[96,256,256]`, `[192,128,128]`, and
`[384,64,64]` condition maps.  They are added only after the official entry,
downsample, or upsample stage boundaries, with five zero-initialized gates.
The core therefore remains the public DiC-S topology rather than a new
"DiC-like" network.

## PixelU-S-BEV correspondence

Reference: `gzp6688/PixelU`, commit
`3b7733d931cbce72cf5f4c5e1ccb0c523d9b65c6`, `model_pixelu.py`,
`PixelU-B-16` / `UiT_B_16`.

| Official component | FPSGen implementation |
| --- | --- |
| `BottleneckPatchEmbed` | Same patch-stride bottleneck then 1x1 lift |
| `JiTBlock` | Same AdaLN, RMSNorm, QK RMSNorm, RoPE SDPA, SwiGLU structure |
| spatial Downsample / Upsample | Same one PixelUnshuffle / PixelShuffle pair |
| shallow-to-deep skips | Same skip on every decoder block |
| topology / patch | `[4,4,4]`, patch-16 unchanged |
| hidden / heads / bottleneck | Width-scaled `768/12/128 -> 384/6/64` |
| class in-context tokens | Replaced by 32 spatial FPSGen condition tokens |
| REPA | Disabled (`repa: false`) |

`PixelU-S-BEV` is an FPSGen width-scaled variant name, **not** an official
PixelU release name.  It preserves the PixelU-B U-shaped topology while
meeting the FPSGen tens-of-millions parameter target.  The adapter produces
256 patch-aligned condition tokens and 32 pooled condition-context tokens,
then injects each through a zero-initialized scalar gate.

## Compatibility and next phase

`build_bev_backbone()` lazy-imports `pixelu_s`.  Checkpoints without
`model.backbone` automatically choose `legacy`; old BEV checkpoints therefore
retain their original `BEVFlowTransNet` state-dict topology.  Inference also
constructs the BEV model from the BEV checkpoint's own hyperparameters.

DiC/PixelU development and future dense benchmarks use
`env/venv_pt20_cu117`.  Inductor is deliberately not activated in code:
future probing may compile only the dense condition adapter/core after eager
PointPillar construction.  The experiment order remains DiC-S eager FP32,
DiC-S real-batch, DiC-S Inductor/AMP, then PixelU-S.

## GPU1 DiC-S training-speed probe (2026-09-29)

This requested engineering probe used the isolated PyTorch 2.0.1/CUDA 11.7
environment and one RTX 3090 (physical GPU1).  It used a fixed real SemanticKITTI
`gt_possion` batch of two full 180k-point clouds, ten warmup optimizer steps,
then fifty timed optimizer steps.  It is **compute-only**: no DataLoader wait or
H2D cost is included.  PointPillar/KeOps KNN, target rasterization and loss stay
FP32; AMP/Inductor apply only after the dense `[B,32,256,256]` LiDAR condition
has been constructed.

| DiC-S mode | Mean ms/step | Samples/s | Peak allocated | Speedup vs FP32 eager | Finite / timed FP16 overflow |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 eager | 329.79 | 6.06 | 8.98 GB | 1.000x | yes / n.a. |
| FP16 eager | 318.05 | 6.29 | 7.60 GB | 1.037x | yes / 0 |
| FP32 Inductor | 325.26 | 6.15 | 8.14 GB | 1.014x | yes / n.a. |
| FP16 Inductor | **271.28** | **7.37** | **5.49 GB** | **1.216x** | yes / 0 |

The dense DiC path does benefit from the combined FP16+Inductor mode.  Inductor
alone is small at the complete-step level because the eager FP32 PointPillar and
KeOps condition frontend remain outside the compile boundary.  These numbers
are a speed probe only, not a training-equivalence result; DiC must still pass
its eager correctness smoke before any formal model run.

## GPU1 Legacy BEVFlow training-speed probe (2026-09-29)

The historical `BEVFlowTransNet` was re-run in the PyTorch 2.0.1/CUDA 11.7
environment using the same `gt_possion` batch and exact 10-warmup/50-timed-step
protocol.  Its prior short legacy-environment measurement is not used here,
because it had a different environment and timing protocol.

| Legacy mode | Mean ms/step | Samples/s | Peak allocated | Speedup vs FP32 eager | Finite / timed FP16 overflow |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 eager | 236.34 | 8.46 | 2.53 GB | 1.000x | yes / n.a. |
| FP16 eager | 220.27 | 9.08 | 2.09 GB | 1.073x | yes / 0 |
| FP32 Inductor | 274.00 | 7.30 | 2.48 GB | 0.863x | yes / n.a. |
| FP16 Inductor | **171.44** | **11.67** | **1.75 GB** | **1.378x** | yes / 0 |

Legacy benefits materially only from the combined FP16+Inductor mode; FP32
Inductor alone regresses on this core.

## GPU1 PixelU-S training-speed probe (2026-09-29)

PixelU-S used the same fixed `gt_possion` batch, RTX 3090, ten warmup optimizer
steps and fifty timed optimizer steps as DiC-S.  This is likewise compute-only:
the PointPillar/KeOps frontend, target rasterization and loss remain FP32 and
eager; AMP and Inductor cover only the dense condition adapter and PixelU core.

| PixelU-S mode | Mean ms/step | Samples/s | Peak allocated | Speedup vs FP32 eager | Finite / timed FP16 overflow |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 eager | 210.01 | 9.52 | 1.03 GB | 1.000x | yes / n.a. |
| FP16 eager | 236.58 | 8.45 | 1.02 GB | 0.888x | yes / 0 |
| FP32 Inductor | **124.74** | **16.03** | 1.02 GB | **1.684x** | yes / n.a. |
| FP16 Inductor | 157.66 | 12.69 | **1.00 GB** | 1.332x | yes / 0 |

For this PyTorch 2.0 / RTX 3090 probe, PixelU-S should use **FP32 + Inductor**:
FP16 is finite and scaler-stable but slower, while Inductor alone supplies the
material speedup.  PixelU's patch-16 tokenization makes it much cheaper than
raw-pixel DiC-S, but this is an engineering comparison rather than a claim of
equal generation quality.  Both backbones still require a correctness smoke and
the same BEV evaluation protocol before a training decision.

## Unified three-backbone comparison

All entries below are directly comparable: physical GPU1 (RTX 3090), PyTorch
2.0.1/CUDA 11.7, a fixed real `gt_possion` batch (`B=2`, 180k full / 18k partial
points), ten warmup optimizer steps, and fifty timed compute-only steps.  The
point geometry, PointPillar/KeOps condition frontend, BEV target construction,
and loss stay eager FP32 for every row; only the dense backbone and its
condition adapter are AMP/Inductor candidates.

| Backbone | Backend | Mean ms/step ↓ | Median / P95 ms | Samples/s ↑ | Peak allocated ↓ | Speedup within backbone | Finite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy | FP32 eager | 236.34 | 212.00 / 307.65 | 8.46 | 2.53 GB | 1.000x | yes |
| Legacy | FP16 eager | 220.27 | 212.60 / 282.99 | 9.08 | 2.09 GB | 1.073x | yes |
| Legacy | FP32 Inductor | 274.00 | 272.42 / 280.98 | 7.30 | 2.48 GB | 0.863x | yes |
| Legacy | **FP16 + Inductor** | **171.44** | 165.02 / 199.20 | **11.67** | **1.75 GB** | **1.378x** | yes |
| DiC-S | FP32 eager | 329.79 | 329.73 / 332.27 | 6.06 | 8.77 GB | 1.000x | yes |
| DiC-S | FP16 eager | 318.05 | 315.73 / 330.10 | 6.29 | 7.42 GB | 1.037x | yes |
| DiC-S | FP32 Inductor | 325.26 | 311.34 / 358.54 | 6.15 | 7.95 GB | 1.014x | yes |
| DiC-S | **FP16 + Inductor** | **271.28** | 274.44 / 287.01 | **7.37** | **5.36 GB** | **1.216x** | yes |
| PixelU-S | FP32 eager | 210.01 | 209.31 / 215.72 | 9.52 | 1.01 GB | 1.000x | yes |
| PixelU-S | FP16 eager | 236.58 | 239.47 / 248.51 | 8.45 | 0.99 GB | 0.888x | yes |
| PixelU-S | **FP32 + Inductor** | **124.74** | 125.66 / 149.17 | **16.03** | 0.99 GB | **1.684x** | yes |
| PixelU-S | FP16 + Inductor | 157.66 | 155.72 / 162.70 | 12.69 | **0.98 GB** | 1.332x | yes |

The per-backbone recommended speed candidates are therefore Legacy
FP16+Inductor, DiC-S FP16+Inductor, and PixelU-S FP32+Inductor.  These are
throughput/memory measurements only: the models have different spatial
topologies and have not yet undergone a generation-quality comparison.
