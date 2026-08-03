# Validation summary

The released code was validated on an NVIDIA GeForce RTX 3090 with Python 3.9,
PyTorch 1.13.0 + CUDA 11.7, MinkowskiEngine 0.5.4, and PyTorch Lightning 1.8.1.

## Pipeline checks

| Check | Result |
| --- | --- |
| Chamfer CUDA operator | Forward/backward check passed |
| BEV smoke training | One real batch completed and checkpoint written |
| Teacher smoke training | One real batch completed and checkpoint written |
| Student smoke training | Teacher loaded; one real batch completed |
| Inference smoke test | 180,000 finite XYZ points generated |
| Eight-condition generation | All eight PLY samples and manifest written |

Dataset alignment was also checked with frame-indexed poses, point-aligned
labels, and fixed 18,000/180,000 input/target cardinalities.

## Completion check

The final local completion run used SemanticKITTI sequence 08, condition `100`,
41 frames sampled every 100 frames, and BEV/PointFlow steps `10/1`.

| Metric | Result |
| --- | --- |
| Chamfer distance | `0.330378 m` (std `0.112203 m`) |
| Completion IoU at 0.5 / 0.2 / 0.1 m | `0.430485 / 0.308478 / 0.174546` |
| 3D JSD | `0.499620` |
| BEV JSD | `0.327917` |

This is a local reproducibility check, not an official benchmark table. Re-run
the final protocol after installing the public environment and record the exact
checkpoint hashes and random seeds.
