# Dense Fidelity V2 GPU Benchmark Status

## Admission result

No GPU benchmark was started. In the current shell:

- `nvidia-smi --query-gpu=...` fails because it cannot communicate with the NVIDIA driver.
- `/dev/nvidia*` does not exist.
- The project Stage1 environment reports PyTorch `1.13.0`, CUDA build `11.7`, and `torch.cuda.is_available() == False`.
- That environment also lacks `torch_scatter`, so importing the real VoxDet ResNet3D module fails before model construction.

This is a runtime environment block, not a model or kernel failure. No GPU task
was launched or interrupted.

## Recorded versus new performance

The V1 historical Encoder-only timings remain reference values and are not
measurements of E1/E2/E3:

| Variant | Historical Encoder mean |
|---|---:|
| R0 Original | 178.26 ms |
| R1 Bottleneck-64 | 138.37 ms |
| R2 OMD-3D | 173.58 ms |
| E1 Bottleneck-96 | NOT TESTED |
| E2 Asymmetric 3D rank-64 | NOT TESTED |
| E3 Lightweight Skip | NOT TESTED |

The earlier random-weight microbenchmark is also not a substitute for new
measurements. It reported R0 203.78 ms, R1 160.14 ms, and R2 195.16 ms at
batch 1 in AMP on physical GPU2. No E1/E2/E3 timing is inferred from these
values.

## Ready-to-run GPU commands

Once the Stage1 Python environment has its pinned dependencies and GPU2/3 are
admitted, use the existing microbenchmark with real frame 00/000100 and the
registered batch/precision protocol:

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2
CUDA_VISIBLE_DEVICES=2,3 /data-12/M2024-HWZ/FPSGen/.conda-envs/fpsgen/bin/python \
  scripts/research_v2/hier_occ_flow/benchmark_dense_encoder_variants.py \
  --physical-gpu 2 --batches 1 2 \
  --variants r0_original e1_bottleneck96 e2_asymmetric3d e3_lightweight_skip \
  --precisions amp_fp16 explicit_fp16 --warmups 20 --repeats 100 \
  --backward --output experiments/dense_fidelity_v2/block_microbench_gpu2.json

CUDA_VISIBLE_DEVICES=2,3 /data-12/M2024-HWZ/FPSGen/.conda-envs/fpsgen/bin/python \
  scripts/research_v2/hier_occ_flow/profile_random_dense_encoder.py \
  --physical-gpu 2 --input-npy /data-12/M2024-HWZ/KITTI_Odometry/00/input_/000100.npy \
  --variants original bottleneck96 asymmetric3d lightweight_skip \
  --precision explicit_fp16 --warmups 20 --repeats 100 \
  --output experiments/dense_fidelity_v2/random_encoder_gpu2.json
```

These commands should only be run after resolving the runtime issue and
checking GPU admission. Training remains a separate gate after the full
Encoder benchmark; do not launch the 20K arms based on CPU toy results.
