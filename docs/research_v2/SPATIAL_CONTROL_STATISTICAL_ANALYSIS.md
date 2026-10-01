# Spatial-control short-run statistical analysis

This report re-analyses the completed 1,500-step experiments without
retraining them.  Generation uses the identical 20 SemanticKITTI sequence-08
frames for each pair, Euler-10, and the same generation seeds.  All intervals
are paired frame bootstraps (10,000 resamples; seed `20261001`).  A positive
delta means the named left model minus the named right model; for Height MAE
and Density Mass-TV, a negative delta is better.

## Generation: key paired differences

| Pair | CFG | IoU delta (95% CI) | Completion F1 delta (95% CI) |
|---|---:|---:|---:|
| S0 SynFlow SPADE - U0 generic U-Net | 1 | +0.01855 [0.01228, 0.02553] | +0.01594 [0.01034, 0.02218] |
| S0 SynFlow SPADE - U0 generic U-Net | 2 | +0.01494 [0.00865, 0.02102] | +0.01347 [0.00742, 0.01924] |
| S0 SynFlow SPADE - C0 decoder-only SPADE | 1 | -0.00101 [-0.00783, 0.00586] | -0.00127 [-0.00725, 0.00472] |
| S0 SynFlow SPADE - C0 decoder-only SPADE | 2 | +0.00312 [-0.00736, 0.01465] | +0.00280 [-0.00685, 0.01319] |
| C0 decoder-only SPADE - U0 generic U-Net | 1 | +0.01956 [0.01064, 0.02890] | +0.01721 [0.00919, 0.02560] |
| C0 decoder-only SPADE - U0 generic U-Net | 2 | +0.01182 [0.00275, 0.02089] | +0.01067 [0.00238, 0.01906] |
| P1 PixelControl - P0 PixelDiT generic | 1 | +0.03755 [0.02093, 0.05776] | +0.03412 [0.01856, 0.05233] |
| P1 PixelControl - P0 PixelDiT generic | 2 | +0.00720 [0.00227, 0.01242] | +0.00804 [0.00389, 0.01236] |

The first result is the robust finding needed for the next phase: S0 is
already better than U0 at CFG=1.  Therefore its gain is not merely a
classifier-free-guidance amplification effect.  S0 versus C0 is not resolved
by B20: both confidence intervals include zero.  They should be treated as an
engineering trade-off, not as a meaningful 0.002 ranking.

## Condition usage: LiDAR-only `Gshuffle`

The B100 evaluator fixes GT, noise, and time per frame, then replaces the
condition with a deterministic wrong frame (`i -> i+17`).  Results below are
paired B100 bootstraps of the per-frame `Gshuffle` values.

| Pair | `t<0.2` delta (95% CI) | `t<0.4` delta (95% CI) |
|---|---:|---:|
| S0 - U0 | +0.05848 [0.05625, 0.06073] | +0.07473 [0.07230, 0.07702] |
| S0 - C0 | +0.01781 [0.01658, 0.01907] | +0.02676 [0.02573, 0.02776] |
| C0 - U0 | +0.04067 [0.03867, 0.04273] | +0.04797 [0.04616, 0.04977] |
| P1 - P0 | -0.00766 [-0.00947, -0.00590] | -0.00358 [-0.00460, -0.00258] |

The full raw analysis, including modes `010`, `001`, `111`, all five time
bins, `Delta Lshuffle`, and `Gshuffle`, is saved in
`outputs/research_v2/spatial_control/spatial_control_paired_bootstrap.json`.

## CFG and efficiency interpretation

Increasing CFG from 1 to 2 improves Completion F1 by 0.01984 (U0), 0.01736
(S0), and 0.01330 (C0).  The S0 advantage is present before that increase,
and does not depend on a disproportionate CFG amplification.

All three U-Net variants have 32.77M total trainable parameters.  The measured
short-run median step time / throughput / reserved memory were: U0 0.3065 s /
26.10 samples/s / 9.68 GB; S0 0.4122 s / 19.41 samples/s / 11.44 GB; C0
0.3329 s / 24.03 samples/s / 10.68 GB.  Thus C0 remains materially cheaper
than S0 while their B20 generation difference is statistically unresolved.

This analysis is exploratory short-run evidence only; it does not replace the
two-seed NCSN++ study or authorize formal long training.
