# DCD-only Teacher: B100 interim evaluation (epoch 01)

## Scope

This is an **interim** evaluation of
`teacher_dcd_a1_l1_5ep_ddp2_bs2_epoch=01.ckpt`, while its five-epoch training
continues. It is not a final CD+rep versus DCD decision.

The fixed seq08 B100 manifest and the historical CD+rep cache are used. For
every one of 100 frames, DCD and CD+rep have bit-identical `P0` and GT. Both
receive exactly the same refinement:

```text
K=16, epsilon=0.01, iterations=100, alpha=1.0
```

Artifacts:

```text
outputs/research_v2/dcd_teacher/cache_b100_epoch01/
outputs/research_v2/dcd_teacher/b100_distribution_epoch01/{results.csv,results.json,summary.json}
```

## Mean B100 geometry

| Method | Chamfer ↓ | P→GT ↓ | GT→P ↓ | F-score ↑ | Coverage ↑ | NN-target coverage ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CD+rep | 0.139568 | 0.144868 | 0.134268 | 0.860048 | 0.885390 | 0.639253 |
| DCD epoch01 | **0.133917** | **0.133083** | 0.134751 | **0.878328** | 0.871404 | 0.602035 |
| CD+rep + Sinkhorn | **0.069757** | **0.072734** | **0.066780** | 0.986957 | 0.990030 | **0.776680** |
| DCD epoch01 + Sinkhorn | 0.072496 | 0.075217 | 0.069775 | **0.987791** | **0.990369** | 0.763747 |

Raw DCD lowers Chamfer in 100/100 frames. With fixed Sinkhorn, CD+rep lowers
Chamfer in 100/100 frames; DCD's slightly higher F-score occurs in 94/100
frames, so F-score alone is not enough to select it.

## Local-density and assignment diagnostics

| Method | Exact duplicate rows ↓ | NN<1cm ↓ | NN<2cm ↓ | NN<5cm ↓ | 8-NN CV ↓ | Assignment count≥2 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CD+rep | 0.00 | 0.0120% | 0.0830% | 0.9864% | **0.273019** | 0.264204 |
| DCD epoch01 | 0.00 | 0.0417% | 0.3188% | 4.3640% | 0.303610 | **0.259728** |
| CD+rep + Sinkhorn | 19.51 | 1.1509% | **2.5067%** | **9.6962%** | **0.282537** | **0.210312** |
| DCD epoch01 + Sinkhorn | **10.87** | **1.0849%** | 2.7004% | 11.6766% | 0.287303 | 0.213451 |

DCD+Sinkhorn reduces exact duplicates on 88/100 frames and reduces NN<1cm on
36/100 frames, but worsens NN<2cm and NN<5cm on 84/100 and 100/100 frames,
respectively. Therefore the observed duplicate reduction is not yet evidence
of a generally healthier local distribution.

## Height distribution

Height-histogram TV improves from `0.025297` (CD+rep+Sinkhorn) to `0.012927`
(DCD+Sinkhorn), in 97/100 frames. This is promising but does not outweigh the
current Chamfer and NN<2/5cm regressions.

## Interim decision

**HOLD.** Preserve the final five-epoch checkpoint and rerun this exact B100
protocol before deciding whether DCD-only replaces CD+rep. Do not start a
Student experiment from this epoch-01 result.
