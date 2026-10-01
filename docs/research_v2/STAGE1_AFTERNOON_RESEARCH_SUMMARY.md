# FPSGen Stage-1 afternoon research summary

**Status:** all scheduled short-run experiments and evaluations are complete.
GPU2/GPU3 are idle.  Formal 500-epoch training was **not started**.

This is a compact evidence packet intended for an independent research review.
Raw condition records, per-frame generation metrics, paired bootstraps, and
two-seed summaries are in `outputs/research_v2/{spatial_control,ncsnpp_spade}`.

## 1. Fixed experimental contract

All results below retain the Stage-1 objective and representation:

- BEV target `[D,H,M]`, shape `3 x 256 x 256`.
- Linear Flow Matching: `Bt=(1-t)B0+tB1`, `u=B1-B0`, direct velocity output.
- Uniform `t in [0,1]`, loss-channel weights `[1,2,1]`.
- Same DynamicKNNPillar LiDAR frontend, vehicle/road layout inputs, GT,
  train split, AdamW (`lr=1e-4`), FP32 eager execution, and effective batch 8.
- Eight equally sampled condition states (`000` through `111`).
- Conditions are evaluated by Correct / Zero / deterministic wrong-frame
  shuffle (`frame i -> i+17`) with the same GT, `B0`, `Bt`, and time.

`Gshuffle=(Lshuffle-Lcorrect)/(Lshuffle+eps)`: higher means the correct
condition lowers velocity loss more than a mismatched condition.  It is an
*usage* metric, not a generation-quality metric.

### Historical original-FPSGen reference (not a paired short-run comparison)

The original FPSGenBEV / legacy BEVFlowTransNet has a completed matched
five-epoch run: the epoch means were `0.7533, 0.1698, 0.1473, 0.1279, 0.1291`.
An earlier five-epoch NCSNpp-S C3/separate run reached `0.3737, 0.1427,
0.1322, 0.1241, 0.1168`.  The early optimization advantage is real in that
historical matched setup, but it is **not** a causal conditioning comparison:
the backbone and condition adapter both differ, and it used a different
five-epoch protocol from the 1,500-step N0/N1 paired study below.  The full
500-step loss curve and caveat are retained in
`BEV_CONDITION_INJECTION_STUDY.md` (section “NCSNpp-S versus original
FPSGenBEV”).

## 2. Spatial-control screen (one training seed, 1,500 steps)

| Model | Final 200 FM loss | LiDAR Gshuffle `<0.2` | B20 IoU CFG2 | B20 Completion F1 CFG2 |
|---|---:|---:|---:|---:|
| U0: generic U-Net additive condition | **0.17306** | 0.24931 | 0.71916 | 0.77692 |
| S0: symmetric SynFlow-style Lite-SPADE | 0.17358 | **0.30794** | **0.73409** | **0.79038** |
| C0: decoder-only SPADE | 0.17513 | 0.28992 | 0.73098 | 0.78759 |
| P0: PixelDiT generic | 0.53931 | 0.04626 | 0.66862 | 0.74356 |
| P1: PixelControl-style PixelDiT | 0.59146 | 0.03865 | 0.67582 | 0.75160 |

### Statistical re-analysis of the spatial screen

B20 uses common frames; 95% CIs are 10,000 paired-frame bootstraps, seed
`20261001`.

| Pair | CFG | IoU delta [95% CI] | Completion F1 delta [95% CI] |
|---|---:|---:|---:|
| S0 - U0 | 1 | +0.01855 [0.01228, 0.02553] | +0.01594 [0.01034, 0.02218] |
| S0 - U0 | 2 | +0.01494 [0.00865, 0.02102] | +0.01347 [0.00742, 0.01924] |
| S0 - C0 | 1 | -0.00101 [-0.00783, 0.00586] | -0.00127 [-0.00725, 0.00472] |
| S0 - C0 | 2 | +0.00312 [-0.00736, 0.01465] | +0.00280 [-0.00685, 0.01319] |

S0 therefore beats U0 before CFG amplification.  The observed S0/C0 gap is
not distinguishable with B20.  PixelControl improved P0 generation metrics in
that particular screen, but its loss and LiDAR usage remained much worse than
the U-Net family; it did not resolve PixelDiT's main optimization/control
bottleneck.

For spatial control usage, the B100 paired difference S0-U0 in LiDAR
`Gshuffle` was +0.05848 [0.05625, 0.06073] at `t<0.2` and +0.07473
[0.07230, 0.07702] at `t<0.4`.  S0-C0 was also positive for usage, but their
generation difference was unresolved.

## 3. NCSNpp generic versus symmetric SPADE (two paired train seeds)

N0 is the existing NCSNpp-S implementation, unchanged: its actual shared
condition pyramid, encoder-side generic spatial additions, and existing
global condition-to-time path are retained.

N1 keeps the exact NCSN++ core, same 34-channel input, same condition encoder
and pyramid, and same global path.  Only the generic spatial additive route
is replaced with symmetric Lite-SPADE at the two primary normalization sites
of every BigGAN ResBlock in encoder, middle, and decoder.  `Delta-gamma` and
`beta` heads are zero initialized; attention is unchanged; no boundary gate.

| Variant | Core params | condition params | total params | median step | reserved memory |
|---|---:|---:|---:|---:|---:|
| N0 generic | 22.809M | 1.642M | 24.453M | 0.402 s | 14.08 GB |
| N1 SPADE | 22.809M | 1.929M | 24.739M | 0.618 s | 19.30 GB |

N1 adds only 286,368 parameters (1.17%), but costs about 54% more median step
time and 5.23 GB more reserved memory.  Median data-ready gaps were 3.8--5.6
ms, far below the 0.40--0.62 s step time; the observed slowdown is model
compute, not a loader bottleneck.

### Dynamics and results

The loss table uses actual per-step CSV records, not sparse TensorBoard
samples.  `1000--1499` is the exact trailing 500-step mean at step 1500.

| Metric | N0 seed42 | N0 seed123 | N1 seed42 | N1 seed123 | N0 mean +/- sd | N1 mean +/- sd |
|---|---:|---:|---:|---:|---:|---:|
| Loss 1000--1499 ↓ | 0.18861 | 0.18885 | 0.18599 | 0.18760 | 0.18873 +/- 0.00012 | **0.18679 +/- 0.00081** |
| Final 200 loss ↓ | 0.17986 | 0.18194 | 0.17726 | 0.18058 | 0.18090 +/- 0.00104 | **0.17892 +/- 0.00166** |
| LiDAR Gshuffle `<0.2` ↑ | 0.28414 | 0.26159 | 0.28213 | 0.27147 | 0.27287 +/- 0.01128 | **0.27680 +/- 0.00533** |
| LiDAR Gshuffle `<0.4` ↑ | 0.22832 | 0.21433 | 0.23376 | 0.22322 | 0.22132 +/- 0.00700 | **0.22849 +/- 0.00527** |
| IoU CFG1 ↑ | 0.69851 | 0.69653 | 0.69575 | 0.69742 | **0.69752 +/- 0.00099** | 0.69659 +/- 0.00083 |
| Completion F1 CFG1 ↑ | 0.76047 | 0.75977 | 0.75805 | 0.76004 | **0.76012 +/- 0.00035** | 0.75905 +/- 0.00100 |
| IoU CFG2 ↑ | 0.71464 | 0.71264 | 0.71232 | 0.70916 | **0.71364 +/- 0.00100** | 0.71074 +/- 0.00158 |
| Completion F1 CFG2 ↑ | 0.77223 | 0.77112 | 0.77037 | 0.76744 | **0.77167 +/- 0.00055** | 0.76890 +/- 0.00146 |

Mean and paired inference, not any single seed's raw value, are the decision
inputs.

### N1-N0 paired bootstrap, B20 generation

Positive values mean N1 minus N0.  Two seed-specific frame-paired bootstrap
results demonstrate that IoU/Completion changes are not consistently positive:

| Metric | seed42 CFG1 | seed42 CFG2 | seed123 CFG1 | seed123 CFG2 |
|---|---:|---:|---:|---:|
| IoU delta | -0.00275 [-0.00582, +0.00053] | -0.00232 [-0.00565, +0.00109] | +0.00089 [-0.00151, +0.00319] | **-0.00349 [-0.00625, -0.00081]** |
| Completion F1 delta | -0.00242 [-0.00522, +0.00056] | -0.00186 [-0.00490, +0.00119] | +0.00028 [-0.00182, +0.00232] | **-0.00368 [-0.00614, -0.00123]** |

N1 does consistently improve height MAE and density Mass-TV in all four
seed/CFG slices (for example seed42 CFG2: -0.0794 m Height MAE and -0.0164
Mass-TV), but those gains did not yield better occupancy or completion.

### Condition-use detail

Entries show `Gshuffle(t<0.2) / Gshuffle(t<0.4)`.

| Seed | Model | `100` LiDAR | `010` vehicle | `001` road | `111` all |
|---:|---|---:|---:|---:|---:|
| 42 | N0 | 0.2841 / 0.2283 | 0.0031 / 0.0025 | 0.1390 / 0.1163 | 0.3690 / 0.3009 |
| 42 | N1 | 0.2821 / 0.2338 | 0.0007 / 0.0006 | 0.1025 / 0.0777 | 0.3706 / 0.3014 |
| 123 | N0 | 0.2616 / 0.2143 | 0.0043 / 0.0033 | 0.1180 / 0.0937 | 0.3431 / 0.2794 |
| 123 | N1 | 0.2715 / 0.2232 | 0.0011 / 0.0010 | 0.0952 / 0.0699 | 0.3418 / 0.2754 |

For `100`, N1-N0 is +0.00713 [0.00576, 0.00848] at `t<0.4` for seed42 and
+0.00945 [0.00834, 0.01057] for seed123.  At `t<0.2`, seed42 is inconclusive
(-0.00132 [-0.00361, +0.00103]) and seed123 is positive (+0.01064
[+0.00887, +0.01250]).  N1 does **not** enhance road or vehicle utilization:
road becomes lower and vehicle remains near zero.

## 4. Evidence-supported reading

1. **Backbone family:** the U-Net-style multiscale models optimize far more
   readily than the current Small PixelDiT under the fixed Flow Matching
   contract.
2. **Spatial conditioning can matter:** S0's better LiDAR usage is accompanied
   by robust B20 improvements over U0, including CFG1.  This is the strongest
   positive result from the afternoon screen.
3. **NCSN++ generic conditioning is already strong:** it achieves very high
   LiDAR usage (`~0.27` early time) and strong completion at this short-run
   budget.
4. **NCSN++ symmetric SPADE is not promoted by present evidence:** it slightly
   lowers FM loss and raises LiDAR `t<0.4` usage, but it lowers/does not
   reproducibly raise IoU and Completion F1, decreases road usage, and has a
   substantial compute/memory cost.  Its status is **HOLD**, not a primary
   formal-training candidate.
5. **Metric distinction is essential:** low FM loss, condition utilization,
   and generation quality do not rank methods identically.  No conclusion
   should choose a long-run candidate from FM loss alone.

## 5. Suggested independent-review question

> Under the fixed Stage-1 Flow Matching contract, should the next experiment
> invest in longer training for N0 generic NCSNpp-S, S0 symmetric-SPADE U-Net,
> or neither?  Consider that S0 has only one training seed but clear paired B20
> gains over U0; N0 has two seeds and is more compute-efficient than N1; and
> N1 gives slightly better loss/partial LiDAR usage but no reliable occupancy
> or completion improvement.  Do not recommend formal long training unless
> the evidence justifies it; suggest the smallest, most informative next test.

## 6. Reproducibility artifacts

- `outputs/research_v2/spatial_control/spatial_control_paired_bootstrap.json`
- `outputs/research_v2/ncsnpp_spade/final_summary.csv`
- `outputs/research_v2/ncsnpp_spade/seed_summary.json`
- `outputs/research_v2/ncsnpp_spade/paired_bootstrap.json`
- Per-frame B20 CSV and B100 records sit beside each experiment output.

No checkpoints, datasets, TensorBoard runs, or generated visual artifacts are
included in version control.
