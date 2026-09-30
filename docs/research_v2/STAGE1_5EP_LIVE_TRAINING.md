# Stage-1 five-epoch BEV training: live loss snapshot

Snapshot date: 2026-10-01.  HDiT-S, DiP-S, and NCSNpp-S are active five-epoch
eager-FP32 runs.  This document is a reproducible progress snapshot, not a
generation-quality comparison.

## Measurement contract

- Scalar: Lightning `train/loss_mse`, the existing weighted Flow-Matching
  velocity MSE using channel weights `[1, 2, 1]`.
- Window: every row averages the five scalar samples emitted at 100-step
  cadence within a contiguous 500 optimizer-step interval.
- Shared data/objective: SemanticKITTI sequences `00–07,09,10`, `gt_possion`,
  180,000 points/frame, eight uniform LiDAR/vehicle/road condition states and
  learning rate `1e-4`.
- Comparability caveat: DiC-S used batch size 2; PixelU-S used batch size 8
  with Inductor.  All three active runs use batch size 8 and eager FP32.

## 500-step mean training loss

| Step | Legacy B8 | DiC-S B2 | PixelU-S B8 Inductor | HDiT-S B8 | DiP-S B8 | NCSNpp-S B8 |
|---|---:|---:|---:|---:|---:|---:|
| 0–499 | 1.6711 | 1.3200 | 1.6370 | 1.3376 | 1.6398 | 1.0491 |
| 500–999 | 1.0084 | 0.3052 | 1.3980 | 0.4761 | 0.5660 | 0.2041 |
| 1,000–1,499 | 0.3966 | 0.1820 | 1.3349 | 0.3669 | 0.4174 | 0.1636 |
| 1,500–1,999 | 0.2352 | 0.1833 | 1.3327 | 0.3435 | 0.4242 | 0.1699 |
| 2,000–2,499 | 0.2256 | 0.1612 | 1.3262 | 0.3423 | 0.4269 | 0.1765 |
| 2,500–2,999 | 0.1900 | 0.1334 | 1.3193 | 0.2665 | 0.3564 | 0.1556 |
| 3,000–3,499 | 0.1775 | 0.1649 | 1.3166 | 0.2559 | 0.3279 | 0.1425* |
| 3,500–3,999 | 0.1480 | 0.1144 | 1.3039 | 0.2490 | 0.2519 | — |
| 4,000–4,499 | 0.1749 | 0.1212 | 1.3096 | 0.2269 | 0.2566 | — |
| 4,500–4,999 | 0.1273 | 0.1355 | 1.3015 | 0.2530 | 0.2011 | — |
| 5,000–5,499 | 0.1781 | 0.1518 | 1.2974 | 0.1979 | 0.2794 | — |
| 5,500–5,999 | 0.1541 | 0.1193 | 1.2934 | 0.2256 | 0.2686 | — |
| 6,000–6,499 | 0.1314 | 0.1179 | 1.2700 | 0.2248 | 0.1984 | — |
| 6,500–6,999 | 0.1491 | 0.1325 | 1.2817 | 0.2103 | 0.2381 | — |
| 7,000–7,499 | 0.1149 | 0.1121 | 1.2805 | 0.2232 | 0.1916 | — |
| 7,500–7,999 | 0.1196 | 0.1055 | 1.2727 | 0.2072 | 0.1962* | — |
| 8,000–8,499 | 0.1345 | 0.1657 | 1.2803 | 0.2177 | — | — |
| 8,500–8,999 | 0.1215 | 0.1239* | 1.2721 | 0.2407 | — | — |

`*` denotes an incomplete 500-step window at snapshot time.  The latest
available optimizer-step scalar was Legacy 11,899; DiC-S 8,799; PixelU-S
18,599; HDiT-S 9,399; DiP-S 7,899; NCSNpp-S 3,099.

## Active-run operational state

The three active configurations use pinned memory, eight deterministic
workers, four-batch prefetching, non-blocking CUDA transfer and a
1,000-step visualization cadence.  Their synchronized profiling CSVs under
`outputs/research_v2/bev_train/profiles/` show p95 data-ready gaps of only a
few milliseconds, far below their p95 optimizer-step times.  They therefore
do not support a DataLoader/I/O bottleneck diagnosis or a large NPY cache.

The paired 16-step preflights passed finite loss, finite gradients, active
condition gradients and batch size 8 before the formal sessions were started.
