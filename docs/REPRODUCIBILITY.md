# From-scratch smoke validation

Run this before a full training job. It uses one sequence, one batch, one epoch
and creates new checkpoints; no pretrained weight is required.

```bash
export CUDA_VISIBLE_DEVICES=0
export TRAIN_DATABASE=/path/to/KITTI_Odometry
python scripts/check_environment.py
python scripts/verify_cuda_ops.py
python -m fpsgen.train_bev --config configs/smoke_bev.yaml
python -m fpsgen.train_teacher --config configs/smoke_teacher.yaml
```

Set `train.teacher_checkpoint` in `configs/smoke_student.yaml` to the teacher
checkpoint just created under `experiments/smoke_teacher/`, then run:

```bash
python -m fpsgen.train_student --config configs/smoke_student.yaml
```

To run the corresponding one-batch test path, add `--weights` with the
checkpoint produced by a smoke run and use `--test`.

Each command must finish with a Lightning checkpoint under `experiments/`. The
student command verifies the teacher checkpoint can be loaded. For a full run,
use the corresponding `configs/train_*.yaml` after setting
`TRAIN_DATABASE`; train the stages in the same order.

To run inference after stage 3, set `FPSGEN_BEV_CHECKPOINT` to a compatible
stage-1 BEV checkpoint and pass the new student checkpoint to
`fpsgen.inference`; the refinement checkpoint is optional. This repository
intentionally does not distribute data or pretrained weights.
