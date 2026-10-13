# Stage1 point occupancy IoU at0.5m — 2026-10-13

The trained network still uses fine256×256×32 cells over
[-50,50]×[-50,50]×[-4,4.4], with physical sizes
0.390625×0.390625×0.2625m. Its coarse128×128×16 grid has
0.78125×0.78125×0.525m cells. No network shape, layer, parameter or
training protocol was changed to obtain the0.5m evaluation.

After the user asked for0.5m voxel IoU, the same formal checkpoint11000 and
same41 seq08 frame IDs0,100,...4000 were used again. The existing density
threshold1.0 was fixed and its metrics exactly reproduced the previous test.
This evaluation also converts predicted density into exactly180000 seeded
Gaussian anchors per frame and measures their binary point occupancy. Only
predicted density generates these sources; GT is used only for evaluation.
PointFlow/Teacher/Sinkhorn are absent.

The cubic metric uses `floor(XYZ/0.5)`, the same floor rule as FPSGen's point
completion IoU, with no morphology closing (radius0). A cell is occupied if
it contains at least one point. The coordinate origin is the LiDAR world
origin, with no centroid or scene AABB recentering. Valid sources/GT keep the
same open50m XY disk and -4<z<4.4 ROI. The upper0.5m Z bin extends to4.5m,
but its point support is restricted to the unchanged4.4m ROI; Z coordinates
are never stretched to force8.4m into an integer number of0.5m cells.

For a resolution control, the exact same180k anchors are also rasterized into
the network's original anisotropic fine cells, again with binary count>0.
Their occupied-cell IDs are identical to allocated_counts>0 because sampling
preserves original source-cell membership. This separates metric resolution
from density-threshold versus point-source binarization.

| Prediction / metric | Per-frame mean IoU | Global confusion-count IoU |
| --- | ---: | ---: |
| Original-grid density, predicted_count>=1 | 44.3503% | 43.8145% |
| Original-grid180k anchors, any point occupied | 40.9270% | 40.7522% |
| **0.5m cubic180k anchors, any point occupied** | **47.3262%** | **47.4535%** |
| 0.5m cubic partial LiDAR points | 17.6940% | 17.2508% |

The same point-cloud occupancy definition improves by6.399 percentage points
when evaluated at0.5m instead of the finer anisotropic network cells.
The44.35% density value has a different binarization and is not a pure
resolution comparison with47.33%. Cubic source IoU varies37.13–54.85% across
the41 scenes; mean F1 at0.5m is64.13%. Global0.5m counts are
TP1326755, FP1020877, FN448271. These are coarse Stage1 source results,
not final PointFlow reconstruction quality or a new0.5m-trained model.

CPU FP32 inference was sharded across two8-thread, low-priority processes to
leave ongoing GPU2+3 training intact. All41 scenes were covered once, with
no missing/duplicate pairs; checkpoint/config/metric protocol identities were
checked. No ROI GT filtering was needed for this set.27 CPU test cases passed
across geometry/point tests and shard merge tests. The point metric has an
independent histogramdd parity test, negative-coordinate floor test, duplicate
point occupancy test and invalid resolution rejection.

Per-frame metrics and full provenance:
`voxel_point_reports/stage1_seq08_voxel05_step11000.json`.
Ignored runtime directory:
`experiments/voxel_point/voxdet_density_formal_gpu23_b2_roi_20261013_094356/seq08_stride100_voxel05_20261013_132250/`.
Each shard saves density, integer allocated counts and180k source anchors;
`merged/per_frame_metrics.csv` contains both resolutions' source IoU.
Add `--point-voxel-size 0.5 --save-density` to the documented stride100
density evaluation command to reproduce this extension. No seq08-driven
threshold, loss-weight or checkpoint selection and no downstream long queue
was performed.
