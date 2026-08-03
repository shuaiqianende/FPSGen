# Dataset format

FPSGen's retained training loader is for preprocessed SemanticKITTI/KITTI
odometry sequences. Set the dataset root with TRAIN_DATABASE, or set
data.data_dir in a training configuration.

Each configured sequence must follow this layout:

```text
KITTI_Odometry/
└── 00/
    ├── calib.txt
    ├── poses.txt
    ├── input_/
    │   ├── 000000.npy
    │   └── ...
    └── gt_/
        ├── 000000.npy
        └── ...
```

gt_/*.npy is the full target point cloud and input_/*.npy is the partial
LiDAR condition. Files must share the same numeric frame names. Each array must
contain at least four columns: the loader reads the first three as XYZ and the
remaining column(s) as point-aligned labels. A numeric frame stem selects the
matching row from that sequence's poses.

The configured data.num_points is the expected full-cloud count. The default
configuration uses 180,000 full points and 18,000 partial points. Preprocess all
samples to fixed cardinalities because the DataLoader stacks arrays without
padding. calib.txt and poses.txt are required because the loader associates
each frame with its pose.

This repository includes preprocessing scripts for rebuilding these arrays from
raw scans and aggregated map chunks. Prepare and validate `input_` and `gt_`
before starting a training run.

## Rebuilding maps and fixed-cardinality arrays

The preprocessing pipeline has two explicit stages. Both commands operate only
on the directory supplied by `--data-root`; no machine-specific path is stored
in the repository.

### 1. Aggregate map chunks

LiDiff-style map exporters usually produce global chunks named
`map_clean_*.pcd` under each sequence. If semantic IDs were written into the
PCD red colour channel, they are decoded automatically. The command below
creates `map_clean.npy` with columns `[x, y, z, semantic_label]` in world
coordinates, plus an optional `map_clean.pcd` for visualization:

```bash
python scripts/aggregate_semantickitti_map.py \
  --data-root /path/to/KITTI_Odometry \
  --sequences 00,01,02,03,04,05,06,07,09,10 \
  --voxel-size 0.1 --backend auto
```

Use `--sequences all` to process every numeric sequence directory. The
deterministic NumPy backend works on CPU; `--backend minkowski --device cuda`
uses the installed MinkowskiEngine build for very large maps. Add `--no-pcd`
when only the NumPy map is needed.

### 2. Generate `input_` and `gt_`

After `map_clean.npy` exists, crop it into every sensor frame and prepare the
fixed-size arrays used by the training loader:

```bash
python scripts/prepare_semantickitti.py \
  --data-root /path/to/KITTI_Odometry \
  --sequences 00,01,02,03,04,05,06,07,09,10 \
  --split train --input-points 18000 --full-points 180000 \
  --max-range 50 --seed 42
```

The script loads `calib.txt`, `poses.txt`, `velodyne/*.bin`, optional
`labels/*.label`, and `map_clean.npy`. It applies the same static-object,
range, height, 10 m viewpoint-cell, voxel-downsampling, and exact-cardinality
steps used by the reference preparation. Each output is a float32 NumPy array
`[N,4] = [x, y, z, semantic_label]` in the current LiDAR frame. Missing labels
are represented by zero, which is suitable for test scans without public
semantic annotations. A per-sequence `fpsgen_preprocess.meta.json` records the
arguments, voxel sizes, point counts, and any replacement sampling.

The default `input_`/`gt_` names are consumed by FPSGen training. To prepare the
generation-evaluation spelling instead, use:

```bash
python scripts/prepare_semantickitti.py \
  --data-root /path/to/KITTI_Odometry --sequences 08 --split test \
  --input-dir input --gt-dir gt
```

The evaluation code accepts both `input`/`gt` and `input_`/`gt_`, so duplicating
large arrays is unnecessary. Use `--strict-cardinality` to fail rather than
sample with replacement when a map crop contains fewer points than requested.

For evaluation, keep the two reference protocols separate:

- Generation evaluation reads fixed-cardinality `gt/*.npy` and prepared
  `input/*.npy` files when they are available.
- SemanticKITTI trajectory completion reads raw `velodyne/*.bin` input and
  crops the dense `map_clean.npy` map using the frame pose. This map-crop target
  is not interchangeable with the fixed 180,000-point generation target.
