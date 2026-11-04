# Stage1 seq08/skip10 408-frame results

Original raw-map ground truth; Stage1 NFE2; native parent centers, no resampling.
U1-Adapted is a 20K model; the other two are 100K. Its comparison is not budget matched.

| Model | Exposure | CD m ↓ | IoU .5 ↑ | IoU .2 ↑ | IoU .1 ↑ | JSD BEV ↓ | JSD 3D ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| U1-Adapted | 20K | 0.275854 | 0.451615 | 0.182487 | 0.029088 | 0.332940 | 0.477229 |
| U1+A | 100K | 0.236836 | 0.478999 | 0.201520 | 0.031977 | 0.310773 | 0.455177 |
| U1+A+depth moment | 100K | 0.237104 | 0.481974 | 0.202152 | 0.032042 | 0.311171 | 0.453539 |

## Paired depth-moment minus U1+A

A 95% percentile block bootstrap resamples contiguous 20-frame blocks (last block has 8 frames), 10,000 replicates, seed 20261105. These intervals describe variability within seq08 only, not uncertainty across scenes or independent training seeds.

| Metric | Mean difference | 95% block-bootstrap interval |
|---|---:|---:|
| cd_m | +0.000268 | [-0.000598, +0.001179] |
| iou05 | +0.002975 | [+0.002154, +0.003735] |
| iou02 | +0.000632 | [+0.000135, +0.001100] |
| iou01 | +0.000065 | [-0.000005, +0.000133] |
| jsd_bev | +0.000398 | [-0.000478, +0.001345] |
| jsd_3d | -0.001638 | [-0.002007, -0.001305] |

The evaluation timer in the existing per-frame records does not include all metric computation; it must not be interpreted as end-to-end evaluation latency.
