# Smoke validation

Use the smoke configurations to verify the three-stage pipeline before a full
training run. They use one sequence, one batch, and one epoch.

## Prepare

```bash
export CUDA_VISIBLE_DEVICES=0
export TRAIN_DATABASE=/path/to/KITTI_Odometry

python scripts/check_environment.py
python scripts/verify_cuda_ops.py
```

## Run the stages

```bash
python -m fpsgen.train_bev --config configs/smoke_bev.yaml
python -m fpsgen.train_teacher --config configs/smoke_teacher.yaml
```

Set `train.teacher_checkpoint` in `configs/smoke_student.yaml` to the Teacher
checkpoint written under `experiments/smoke_teacher/`, then run:

```bash
python -m fpsgen.train_student --config configs/smoke_student.yaml
```

Each command should write a Lightning checkpoint under `experiments/`. For a
full run, use the matching `configs/train_*.yaml` files in the same order.

## Test and inference

Pass `--weights /path/to/checkpoint.ckpt --test` to run a one-batch test path.
For inference, set `FPSGEN_BEV_CHECKPOINT` and pass the Student checkpoint to
`fpsgen.inference`. Released checkpoint links are listed in the README.
