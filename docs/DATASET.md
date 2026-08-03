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

This repository intentionally does not include preprocessing code. Prepare and
validate input_ and gt_ before starting a training run.

For evaluation, keep the two reference protocols separate:

- Generation evaluation reads fixed-cardinality `gt/*.npy` and prepared
  `input/*.npy` files when they are available.
- SemanticKITTI trajectory completion reads raw `velodyne/*.bin` input and
  crops the dense `map_clean.npy` map using the frame pose. This map-crop target
  is not interchangeable with the fixed 180,000-point generation target.
