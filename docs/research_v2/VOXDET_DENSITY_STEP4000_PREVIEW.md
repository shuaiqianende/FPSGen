# Stage1 checkpoint4000 inference preview — 2026-10-13

Five actual scenes were inferred from the saved formal VoxDet density
checkpoint4000, then allocated exactly180000 valid-cell/circular-ROI Gaussian
anchors each. Two training scenes:00/000447,05/001000. Three held-out development
scenes:00/000550,05/000550,07/000550. No seq08 was read. Input manifests contain
onlyid/input; GT is loaded only by a separate CPU evaluation/rendering script.
Teacher, Sinkhorn, PointFlow and Stage1 hidden features are absent from this
density-only source preview. This is not a final PointFlow scene-quality result.

The inference GPU3 probe with a1.05GiB allocator cap could not fit alongside
the training allocation; it exited without stopping training. All five actual
previews use the same checkpoint in CPU FP32,8threads,23.43s/scene on average.
For CPU compatibility only, the unchanged official tokenizer forward body is
bound with `points[0].get_device()` replaced by `points[0].device`. No learned
layers, tensor shapes, pooling operations or parameters change. Original/vendor
source remains unchanged. CPU preview timing is not deployment GPU latency.

All point sources are finite float32[180000,3], seeded and inside their assigned
voxels and the circular ROI. GT00/000447 excludes the previously reported one
out-of-ROI point, giving179999 supervised/evaluation points; source allocation
still returns180000. Predicted density sums range182247–188738 across five scenes.

Three-scene development means, with fixed diagnostic occupancy threshold1.0:

| Metric | Partial input density | Predicted density |
| --- | ---: | ---: |
| 3D occupancy IoU | 0.1240 | 0.4397 |
| 3D occupancy F1 | 0.2199 | 0.6103 |
| Occupied-voxel count MAE | 2.2532 | 1.3543 |
| Log-count MAE over all valid voxels | 0.0471 | 0.0585 |
| Normalized density Mass-TV | 0.8386 | 0.5094 |
| Coarse Mass-TV | 0.6832 | 0.3730 |

Predicted total-mass relative error1.619%;180k predicted-density anchors have
symmetric Euclidean CD0.3462m, F-score@10cm0.1472, F-score@20cm0.5780,
GT coverage@20cm0.6462. Oracle GT-density anchors have CD0.1329m and
F-score@20cm0.8937. Even this oracle source has voxel/Gaussian surface blur,
which motivates the independent PointFlow stage.

Visual comparison shows recognizable road/curb support but thicker, diffuse
surfaces, scattered false-positive support, and insufficient tall vertical
geometry. The all-valid-voxel log-count MAE is worse than partial input despite
better occupancy and spatial-mass metrics, so background error remains material.
This is a small early-checkpoint diagnostic, not a full-development evaluation,
threshold selection, frozen test result or evidence that later training will
necessarily resolve every defect. Exact per-frame metrics/config and checkpoint
SHA256s are in `voxel_point_reports/stage1_preview_step4000.json`.

Large arrays and projection images remain under the ignored run's
`preview_20261013_105510/` directory. To reproduce inference, use
`sample_voxdet_density.py --config ... --output ... --input-manifest ... --device cpu`;
the separate `preview_density_results.py` accepts the evaluation manifest and
saved inference root. Formal training remains on physicalGPU2/3 with no restart.
At11:03 local time it had4588/25000 updates; recent1.002s/update gives roughly
7h total and5h41m remaining, ETA16:44 local time. This budget covers Stage1 only.
