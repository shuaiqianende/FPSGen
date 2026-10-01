# Spatial-control 1500-step results

All five runs used FP32 eager, effective batch 8, seed 42, the unchanged FM
objective and optimizer schedule.  Each completed 1,500 optimizer steps; the
results below are short-run evidence only, not a 500-epoch conclusion.

| Model | trailing FM @500 | @1000 | @1500 | Gshuffle100 `<.2` | Gshuffle100 `<.4` | B20 CFG2 IoU | B20 CFG2 Completion F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| U0 generic U-Net | 1.268 | 0.397 | **0.173** | 0.249 | 0.199 | 0.719 | 0.777 |
| S0 symmetric Lite-SPADE | 1.273 | **0.388** | 0.174 | **0.308** | **0.271** | **0.734** | **0.790** |
| C0 decoder-only Lite-SPADE | **1.267** | 0.394 | 0.175 | 0.290 | 0.245 | 0.731 | 0.788 |
| P0 PixelDiT generic | 1.518 | 0.906 | 0.539 | 0.046 | 0.024 | 0.669 | 0.744 |
| P1 PixelControl | 1.487 | 0.941 | 0.591 | 0.039 | 0.021 | 0.676 | 0.752 |

## Interpretation

The shared U-Net core learns substantially faster than the PixelDiT core under
the fixed comparison contract.  S0 has nearly the same FM loss as U0 but a
larger correct-vs-wrong LiDAR gap and the best B20 screen scores.  C0 remains
close behind S0: decoder-only SPADE is effective, but the short-run evidence
favours symmetric encoder/middle/decoder control.

P1 is stable, but does not improve P0's FM loss or LiDAR Gshuffle.  Its B20
IoU/F1 improve slightly while height MAE worsens (2.274m vs P0's 2.080m), so
the zero-adapter route has not yet demonstrated stronger spatial control.

No condition-insensitive veto applies to U0/S0/C0.  Both PixelDiT variants
remain above the 0.02 LiDAR `Gshuffle100(t<0.4)` threshold, but their values
are much lower than the U-Net family and should be treated as weak condition
use.
