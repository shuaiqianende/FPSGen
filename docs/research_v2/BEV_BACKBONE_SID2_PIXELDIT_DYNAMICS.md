# SiD2 / PixelDiT Stage-1 dynamics

The CPU tests and B1/B2/B4/B8 GPU smoke completed before the fixed-seed
five-epoch runs. Both models use eager FP32 and effective batch 8. Condition
usage B100 and LiDAR-only B20 are pending in the post-training evaluation
queue; no formal long training is started by this study.

| Model | Core+condition params | Microbatch | Accumulation | Effective batch | Peak memory | step/s | 500-step FM loss | Gshuffle100 <0.2 | Gshuffle100 <0.4 | B20 IoU | B20 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SiD2-S-BEV | 44.48M | 8 | 1 | 8 | 6,540 reserved MiB | 31.88 | 0.1597 (step 11,500) | pending | pending | pending | pending |
| PixelDiT-S-BEV | 22.79M | 8 | 1 | 8 | 4,908 reserved MiB | 41.08 | 0.2222 (step 11,500) | pending | pending | pending | pending |

## Completed training dynamics

| Model | Epoch means, 0→4 | Final epoch mean | p50 / p95 step (s) | p50 / p95 data gap (s) | Peak allocated MiB |
|---|---|---:|---:|---:|---:|
| SiD2-S-BEV | 0.7465, 0.2087, 0.1836, 0.1706, 0.1653 | 0.1653 | 0.2508 / 0.2702 | 0.0054 / 0.0068 | 756.8 |
| PixelDiT-S-BEV | 0.8410, 0.2952, 0.2603, 0.2385, 0.2284 | 0.2284 | 0.1951 / 0.3906 | 0.0051 / 0.0110 | 408.4 |

The data-ready gap is below 3% of median step time for both runs, so the
measured limitation is model compute, not DataLoader/I/O. SiD2 improves on
the completed HDiT-S and DiP-S loss curves but remains above the historical
NCSNpp-S and original FPSGenBEV baselines. PixelDiT avoids the old PixelU loss
plateau but remains the weakest completed raw-pixel backbone by FM loss here.
