# Isolated PyTorch-2 Teacher speed environment

## Purpose

This directory is an engineering-only environment for verifying PyTorch 2.0.1,
Inductor, FP16/BF16 and MinkowskiEngine 0.5.4 compatibility on physical GPU1.
It must not modify the legacy environment, formal DCD Teacher code, checkpoints,
or GPU2/3 training.

## Baseline and candidate

| Item | Legacy | Candidate |
| --- | --- | --- |
| Python | 3.9 | 3.9 |
| PyTorch | 1.13.0 | 2.0.1 |
| CUDA runtime | 11.7 | 11.7 |
| MinkowskiEngine | 0.5.4 | rebuild 0.5.4 |

Use only `CUDA_VISIBLE_DEVICES=1`; inside Python the only legal device is
`cuda:0`. Coordinates and voxel quantization always remain FP32.

## Install and build

```bash
env/scripts/create_env.sh
source /data-12/M2024-HWZ/FPSGen/env/venv_pt20_cu117/bin/activate
env/scripts/build_minkowski.sh
env/scripts/build_extensions.sh
CUDA_VISIBLE_DEVICES=1 python env/scripts/check_environment.py
```

All source builds use CUDA 11.7, `TORCH_CUDA_ARCH_LIST=8.6`, and fresh local
KeOps cache. Do not copy `.so` files from the legacy environment. ME 0.5.4
needs the documented local source patch at
`env/patches/minkowskiengine-0.5.4-pytorch2-aten-include.patch` when compiled
against PyTorch 2.0.1; it is a header-order fix in `src/spmm.cu`, not an
algorithmic change.

The host has no standalone Python 3.9. The virtualenv lives wholly under
`env/`; it uses the legacy Python 3.9 executable only as an immutable base
interpreter and installs no package into the legacy environment.

## Gates and benchmark order

1. Torch CUDA, dense AMP, dense compile, ME import, Chamfer forward/backward.
2. 100 FP32 ME iterations; then 100 FP16 and BF16 ME iterations.
3. Teacher AMP only after ME passes: coordinates/P0/GT/DCD stay FP32.
4. Dense compile then graph-break analysis; full sparse Teacher is experimental.
5. Cached real 180k batch: three warmup and ten timed steps.

## Verified results (GPU1, RTX 3090)

All real-Teacher measurements use DCD-only, batch=2, 180k points, 3 warmup
steps and 10 timed steps. `P0`, coordinate/TensorField construction, endpoint
arithmetic and DCD stayed FP32; AMP only wrapped the ME backbone.

| Configuration | Mean step | Peak allocated | Result |
| --- | ---: | ---: | --- |
| Legacy Torch 1.13 FP32 | 2077.4 ms | 13.11 GB | baseline |
| Torch 2.0 FP32 | 2101.3 ms | 13.13 GB | no benefit |
| Torch 2.0 FP16 | 1999.9 ms | 11.87 GB | finite; 9.4% less memory |
| Torch 2.0 BF16 | 2003.8 ms | 11.87 GB | finite; 9.4% less memory |
| Torch 2.0 FP32, no hot-path `empty_cache` | 1801.0 ms | 13.13 GB | 1.153x legacy |
| Torch 2.0 BF16, no hot-path `empty_cache` | 1759.6 ms | 11.88 GB | **1.181x legacy; 9.4% less memory** |

ME 0.5.4 completed 100 FP32, FP16 and BF16 forward/backward iterations. The
rebuilt Chamfer3D extension also passed CUDA forward/backward. Dense Inductor
works, but the actual ME Conv/BN/ReLU probe yielded three Dynamo graphs, two
graph breaks and zero captured ops; `torch.compile` is therefore not an
effective sparse-Teacher speed target. See `env/results/*.csv` for the
machine-readable record.

## Dense BEV backbone development

`DiC-S-BEV` and `PixelU-S-BEV` are developed and statically validated in this
PyTorch-2 virtual environment.  Legacy BEVFlow remains compatible with the
PyTorch-1.13 environment because the new backbone factory lazily imports
PixelU only when explicitly selected by configuration.

This preparation phase intentionally does **not** enable Inductor, FP16, or
BF16.  The later speed study must compile only the dense condition adapter and
DiC/PixelU core after the eager FP32 condition frontend has produced its LiDAR
feature map; PyKeOps PointPillar construction is outside that boundary.
