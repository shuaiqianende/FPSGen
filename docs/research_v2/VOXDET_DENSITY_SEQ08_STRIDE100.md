# Interim Stage1 seq08 stride100 test — 2026-10-13

Training uses sequences00–07,09,10 with the previously frozen17870 training,
900 development and360 guard-frame manifest. Seq08 never enters training.
The user explicitly authorized seq08 testing on2026-10-13. The instruction
"every100step" was interpreted as numeric test-frame IDs0,100,200,...4000,
not evaluating every100 optimizer updates. Seq08 has4071 paired scenes and
this manifest contains exactly41. All41 were accepted and evaluated exactly
once; no scene was silently skipped and no ROI point was filtered in this set.

Before loading test targets, the latest available formal checkpoint11000 and
the previously used density occupancy threshold1.0 were frozen in a separate
experiment snapshot. Its protocol file and complete numeric input/GT pairing
manifest accompany the results. No threshold/loss/model/config was selected
using this test. The ongoing25k-update GPU2+3 Stage1 job was not restarted,
stopped or modified. Teacher/Sinkhorn/PointFlow remain outside this test.

Both GPUs were occupied by training, so evaluation used CPU FP32 in two
disjoint shards,8threads each, with reduced process priority and separate CPU
affinities. It uses the same model parameters/architecture; the CPU-only
tokenizer device selection compatibility is documented in the step4000 preview
report. The21/20-scene shards completed in roughly9minutes. The merge checks
checkpoint/config/protocol identity, no duplicated frames, exact pairing and
complete coverage of the frozen41-scene manifest. Macro means are computed
from all41 rows, not an unweighted average of the unequal shards.

| Per-scene mean metric | Partial input density | VoxDet density |
| --- | ---: | ---: |
| 3D occupancy IoU | 0.12361 | 0.44350 |
| 3D occupancy F1 | 0.21917 | 0.61283 |
| Occupied-voxel count MAE | 2.13778 | 1.25173 |
| Log-count MAE over all valid voxels | 0.04935 | 0.05735 |
| Normalized density Mass-TV | 0.84394 | 0.49240 |
| Coarse density Mass-TV | 0.68256 | 0.34970 |

Global micro IoU, aggregating voxel confusion counts across scenes: VoxDet
0.438145, input0.118081. VoxDet TP2056197, FP1236717, FN1400044.
Predicted density mean total189268.9 points, mean relative total-mass error
5.289%. GT180000 per scene; input18000 per scene. Background/log-count error
remains higher than partial input despite better support and distribution
metrics. These are interim Stage1 density results on41 sampled test scenes,
not final25k-checkpoint or PointFlow point-quality results. The prior3-scene
development mean at4000 updates is a different split and cannot establish
a checkpoint-to-checkpoint test-quality improvement.

Complete per-frame metrics, checkpoint/config SHA256s and confusion counts:
`voxel_point_reports/stage1_seq08_stride100_step11000.json`. Ignored runtime
output: `experiments/voxel_point/voxdet_density_formal_gpu23_b2_roi_20261013_094356/seq08_stride100_20261013_125853/`.
Its `merged/per_frame_metrics.csv` provides the41 individual rows.

Reproduce a shard with:

```bash
env/voxdet_density_py39/bin/python scripts/research_v2/voxel_point/eval_voxdet_density.py \
  --config CONFIG --output OUTPUT --split test --frame-stride 100 \
  --device cpu --cpu-threads 8 --num-shards 2 --shard-index 0
```

Then use `merge_density_evaluations.py --input-root ROOT --expected-manifest
test_manifest.json --output MERGED`. The merge tests cover unequal shard
sizes, missing/duplicate scenes, differing checkpoint hashes and wrong pairing;
the original20 voxel/point CPU tests also pass. No new long training queue
was started.
