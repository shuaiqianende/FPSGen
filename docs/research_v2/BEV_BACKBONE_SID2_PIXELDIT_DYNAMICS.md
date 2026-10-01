# SiD2 / PixelDiT Stage-1 dynamics

This report is populated only after CPU tests, GPU smoke, batch-size ladder,
five-epoch training, condition-usage B100, and LiDAR-only B20 evaluations.
No formal long training is started by this study.

| Model | Core+condition params | Microbatch | Accumulation | Effective batch | Peak memory | step/s | 500-step FM loss | Gshuffle100 <0.2 | Gshuffle100 <0.4 | B20 IoU | B20 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SiD2-S-BEV | pending | pending | pending | 8 | pending | pending | pending | pending | pending | pending | pending |
| PixelDiT-S-BEV | pending | pending | pending | 8 | pending | pending | pending | pending | pending | pending | pending |
