# Stage1 + Stage2 joint evaluation, seq08 / skip10

The first arm has completed all 408 frames. Evaluation uses the depth-moment U1+A Stage1 EMA (100K exposures), Stage1 NFE2 and threshold 0.5, Stage2 NFE4 and endpoint threshold zero. Stage2 uses LiDAR Encoder conditioning without Stage1 score. Predicted fine voxel centers are resampled to exactly 1,080,000 points at 0.15625 m, with seed `20261019 + frame`. Ground truth is the original full RawMapSource crop; no ground-truth subsampling. Networks use explicit FP16; Euler and geometry remain FP32.

| Stage2 variant | Frames | CD m ↓ | IoU@0.5 ↑ | IoU@0.2 ↑ | IoU@0.1 ↑ | JSD-BEV ↓ | JSD-3D ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder conditioning | 408 | 0.197932 | 0.509543 | 0.390424 | 0.174089 | 0.303421 | 0.417762 |
| No Encoder | 408 | 0.198513 | 0.509303 | 0.390124 | 0.174016 | 0.303739 | 0.418189 |
| Child position | 408 | 0.197551 | 0.509938 | 0.390358 | 0.173765 | 0.302816 | 0.417608 |
| Late Encoder injection | NOT COMPLETE | — | — | — | — | — | — |

These are macro means across frames. Encoder-conditioned Stage2 checkpoint SHA256 is `67762b5c13a56eeb1596b409aeaddda25712733b67ed3a5ee7a89dde3ad5908e`; no-Encoder Stage2 checkpoint SHA256 is `91e10cfb187ac57ac2c8f9bb6e891dd6a6b67b9ea32a53cb7c5c4ceca99e050b`; the Stage1 EMA checkpoint SHA256 is `f0aaf51a5c666fc374ade4ff62dab507bfcc5ec2164290c008c1bdafcde9b859`. The full runtime identity and per-frame metrics are recorded in the experiment output. The first smoke exposed rejection failure for near-tangent sampling cells; the formal result uses a conservative smaller proposal box with the same exact sphere rejection and no boundary clamping.

## Encoder minus no Encoder: paired differences

The same 408 frame IDs and ground-truth counts were checked. The 95% percentile intervals use 10,000 bootstrap resamples of consecutive 20-frame blocks, seed 20261105 (last block has eight frames). They reflect variability within seq08, not variability across training seeds or independent sequences.

| Metric | Mean difference | 95% block interval |
|---|---:|---:|
| CD m | -0.000581 | [-0.000758, -0.000401] |
| IoU@0.5 | +0.000239 | [+0.000092, +0.000390] |
| IoU@0.2 | +0.000300 | [+0.000238, +0.000365] |
| IoU@0.1 | +0.000073 | [+0.000040, +0.000111] |
| JSD-BEV | -0.000318 | [-0.000407, -0.000226] |
| JSD-3D | -0.000427 | [-0.000485, -0.000370] |

These single-checkpoint comparisons show a small advantage on this sequence; they do not establish stability across independent training runs or other datasets. Other arms remain incomplete.
