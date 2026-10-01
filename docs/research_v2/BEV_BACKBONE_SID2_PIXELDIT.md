# SiD2-S-BEV and PixelDiT-S-BEV

Both additions remain Stage-1 raw 3-channel BEV velocity predictors under the
existing linear Flow Matching objective, uniform continuous time and weighted
`[1,2,1]` FM loss.  They do not introduce a VAE, x-prediction, REPA, mixed
precision or Inductor.

## SiD2-S-BEV

SiD2-S-BEV is a **paper-based Residual U-ViT implementation**, not official
SiD2 source code.  It uses PixelUnshuffle-2 and channel widths
`[64,128,256,384]`, ResBlocks at 128/64, Transformer blocks at 32/16, E3-D3
levels, and 16 middle Transformer blocks.  Its only U-shaped connection is
the level-wise residual `up(low - down) + high`; it has no block-wise
concatenative U-Net skips.

Separate-first bias-free LiDAR/vehicle/road projections make a spatial
pyramid and per-level global modulation.  Both spatial and global gates start
at `0.1`, and exact all-zero conditions remain exactly zero.

## PixelDiT-S-BEV

PixelDiT-S-BEV is architecture-aligned to NVlabs/PixelDiT commit
`41f73006ae532b0b41fee72b181dc22891a5a01a`.  It is a custom scaled variant,
not an official NVIDIA configuration: patch size 16, hidden size 384, six
64-dimensional heads, eight patch DiT blocks, and four PiT blocks with pixel
hidden size 8.  PiT post-AdaLN modulation is enabled.

Conditions are injected after bias-free separate-first cell projection at all
three relevant paths: pixel tokens, patch tokens, and global AdaLN context.
Time remains the FPSGen value in `[0,1]` (`time_scale: 1.0`); no `t*1000` is
applied.
