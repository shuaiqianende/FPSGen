# Formal results: NOT AVAILABLE

No formal HS-OccFlow training was launched; no paper claim is supported yet.
Representation, I/O and CPU mechanics results are separate milestones. GPU smoke,
full-data preprocessing, matched BCE/FM overfit and1K/5K pilots remain required.
Geometry and candidate/endpoint thresholds are not frozen for formal comparison.

## Existing Stage1 candidate ceiling diagnostic

One development frame00/000500, existing input-only VoxDet25K density vs **new raw
map occupancy**. This is not900-frame calibration. The checkpoint was trained on
historical Poisson density; the changed target support is an explicit domain gap.

| Raw count policy | Parents | Precision | Recall | Fine recall ceiling |
|---|---:|---:|---:|---:|
| {'strategy': 'threshold', 'threshold': 0.01} | 176,660 | 0.3742 | 0.8912 | 0.9450 |
| {'strategy': 'threshold', 'threshold': 0.05} | 147,845 | 0.4337 | 0.8645 | 0.9307 |
| {'strategy': 'threshold', 'threshold': 0.1} | 134,385 | 0.4674 | 0.8469 | 0.9207 |
| {'strategy': 'threshold', 'threshold': 0.25} | 114,788 | 0.5275 | 0.8163 | 0.9043 |
| {'strategy': 'threshold', 'threshold': 0.5} | 97,645 | 0.5908 | 0.7778 | 0.8825 |
| {'strategy': 'threshold', 'threshold': 1.0} | 74,581 | 0.6944 | 0.6983 | 0.8350 |
| {'strategy': 'topk', 'top_k': 50000} | 49,999 | 0.8157 | 0.5499 | 0.7291 |
| {'strategy': 'topk', 'top_k': 100000} | 99,995 | 0.5820 | 0.7845 | 0.8862 |
| {'strategy': 'topk', 'top_k': 200000} | 199,982 | 0.3362 | 0.9065 | 0.9533 |
| {'budget': 200000, 'expansion': 1, 'strategy': 'threshold', 'threshold': 0.5} | 181,510 | 0.3558 | 0.8708 | 0.9334 |

At raw-count0.5,11.75% of fine GT support is unreachable by Stage2. At0.01 this
falls to5.50%, with more false-positive parents/tokens. Thresholds cannot be
selected from this one frame, and seq08 was not accessed for this decision.

Next execution order: free allowed GPU admission → actual Stage1/parent PTv3
shapes and memory/permutation → locked geometry → full versioned labels → matched
oracle BCE/FM overfit →1K pilots → only after quality gates5K/predicted-parent work.
No100K queue or Stage2 PointFlow/Teacher/Sinkhorn job has been created.
