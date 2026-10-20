# HS-OccFlow V6 quality regression

The full explicit-FP16 candidate was evaluated on all 41 SemanticKITTI sequence 08 / skip100 frames against the frozen V4 predictions, with the same GT source, ROI, exact GPU KD-tree Chamfer, corrected physical-grid IoU, and shared JSD implementation. No jitter was used for this network-quality check.

| Metric | V4 reference | Full explicit FP16 | Delta |
|---|---:|---:|---:|
| CD (m) | 0.202787 | 0.203333 | +0.000545 m (+0.269%) |
| IoU@0.5 m | 50.6824% | 50.6766% | −0.0058 percentage points |
| IoU@0.2 m | 39.5352% | 39.5471% | +0.0119 percentage points |
| IoU@0.1 m | 16.4191% | 16.4235% | +0.0044 percentage points |
| JSD-BEV | 0.307588 | 0.307825 | +0.000237 |
| JSD-3D | 0.416162 | 0.416187 | +0.000026 |
| Occupied voxels / frame | 613,950 | 614,327 | +376 |

All aggregate changes are within the task's proposed aggregate screening thresholds. It is a numerical variant, not exact parity: Stage1 candidate parents and fine occupancy changed on some frames. The worst CD regression was +0.005634 m at frame 1100; the strongest improvement was −0.002697 m at frame 2200. The worst IoU@0.1 regression was +0.000130 absolute at frame 1300, while the largest decrease was −0.000070 at frame 3700. These per-frame changes are small but should remain visible in the saved per-frame CSV.

This is a 41-frame quality pass against V4; a 408-frame evaluation was not run. Metrics are native occupancy-center metrics, not 1.08M jitter metrics.
