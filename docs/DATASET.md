# Dataset format

FPSGen uses preprocessed SemanticKITTI/KITTI-odometry sequences. Set the data
root with `TRAIN_DATABASE` or `data.data_dir` in a training configuration.

## Sequence layout

```text
KITTI_Odometry/
└── 00/
    ├── calib.txt
    ├── poses.txt
    ├── map_clean.npy
    ├── velodyne/
    ├── input_/
    │   ├── 000000.npy
    │   └── ...
    └── gt_/
        ├── 000000.npy
        └── ...
```

`input_/*.npy` contains the partial LiDAR condition and `gt_/*.npy` contains
the full target. Files must share numeric frame names. Each array is float32
`[N, 4] = [x, y, z, semantic_label]` in the current LiDAR frame. The default
cardinalities are 18,000 input points and 180,000 target points.

## Prepare maps

Aggregate LiDiff-style `map_clean_*.pcd` chunks into a global map:

```bash
python scripts/aggregate_semantickitti_map.py \
  --data-root /path/to/KITTI_Odometry \
  --sequences 00,01,02,03,04,05,06,07,09,10 \
  --voxel-size 0.1 --backend auto
```

The script writes `map_clean.npy` in world coordinates and, unless
`--no-pcd` is supplied, a visualizable `map_clean.pcd`. Labels stored in the
red PCD colour channel are decoded automatically.

## Prepare training arrays

```bash
python scripts/prepare_semantickitti.py \
  --data-root /path/to/KITTI_Odometry \
  --sequences 00,01,02,03,04,05,06,07,09,10 \
  --split train --input-points 18000 --full-points 180000 \
  --max-range 50 --seed 42
```

This creates `input_/*.npy` and `gt_/*.npy` after applying scan filtering,
pose-based map cropping, viewpoint filtering, voxel downsampling, and fixed
cardinality sampling. Each sequence also receives `fpsgen_preprocess.meta.json`.

For the `input`/`gt` spelling used by some generation-evaluation layouts, pass
`--input-dir input --gt-dir gt`. The released evaluators accept both naming
conventions, so datasets do not need to be duplicated.

## Evaluation inputs

- Generation evaluation uses prepared `input[_]/*.npy` and `gt[_]/*.npy`.
- Completion evaluation uses the raw scan and `map_clean.npy`.

These targets are different and should not be interchanged.
