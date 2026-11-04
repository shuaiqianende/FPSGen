# Stage1 + Stage2 joint evaluation, seq08 / skip10

The first arm has completed all 408 frames. Evaluation uses the depth-moment U1+A Stage1 EMA (100K exposures), Stage1 NFE2 and threshold 0.5, Stage2 NFE4 and endpoint threshold zero. Stage2 uses LiDAR Encoder conditioning without Stage1 score. Predicted fine voxel centers are resampled to exactly 1,080,000 points at 0.15625 m, with seed `20261019 + frame`. Ground truth is the original full RawMapSource crop; no ground-truth subsampling. Networks use explicit FP16; Euler and geometry remain FP32.

| Stage2 variant | Frames | CD m ↓ | IoU@0.5 ↑ | IoU@0.2 ↑ | IoU@0.1 ↑ | JSD-BEV ↓ | JSD-3D ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder conditioning | 408 | 0.197932 | 0.509543 | 0.390424 | 0.174089 | 0.303421 | 0.417762 |
| No Encoder | NOT COMPLETE | — | — | — | — | — | — |
| Child position | NOT COMPLETE | — | — | — | — | — | — |
| Late Encoder injection | NOT COMPLETE | — | — | — | — | — | — |

These are macro means across frames. No superiority of Encoder conditioning is established until the other arms finish and a paired analysis is performed. The Stage2 checkpoint SHA256 is `67762b5c13a56eeb1596b409aeaddda25712733b67ed3a5ee7a89dde3ad5908e`; the Stage1 EMA checkpoint SHA256 is `f0aaf51a5c666fc374ade4ff62dab507bfcc5ec2164290c008c1bdafcde9b859`. The full runtime identity and per-frame metrics are recorded in the experiment output. The first smoke exposed rejection failure for near-tangent sampling cells; the formal result uses a conservative smaller proposal box with the same exact sphere rejection and no boundary clamping.
