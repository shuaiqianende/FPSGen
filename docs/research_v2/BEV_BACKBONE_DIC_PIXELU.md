# Stage-1 BEV Backbones: DiC-S-BEV and PixelU-S-BEV

Status: **code ready only**.  No GPU smoke, training, BEV evaluation, Inductor
benchmark, or AMP benchmark is included in this preparation change.

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
