# Installation

FPSGen was validated with Python 3.9, PyTorch 1.13.0, CUDA 11.7, PyTorch
Lightning 1.8.1, and MinkowskiEngine 0.5.4. Build every CUDA extension with
the same Python, PyTorch, CUDA toolkit, and C++ compiler.

## Environment

```bash
conda env create -f environment.yml
conda activate fpsgen
```

Install MinkowskiEngine after PyTorch is available:

```bash
export CUDA_HOME=/usr/local/cuda-11.7
export MAX_JOBS=8
pip install --no-deps MinkowskiEngine==0.5.4
```

## CUDA extensions

Install FPSGen and the bundled CUDA extensions from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
pip install -e fpsgen/models/ChamferDistancePytorch/chamfer3D
pip install -e fpsgen/utils/metrics_gen/pytorch_structural_losses
```

Chamfer distance is required by Teacher training and point-cloud evaluation.
The structural-loss extension is required by `fpsgen.utils.eval_generation`.
The Chamfer implementation is distributed under its upstream MIT license at
`third_party/licenses/ChamferDistancePytorch-MIT.txt`.

## Verification

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py
CUDA_VISIBLE_DEVICES=0 python scripts/verify_cuda_ops.py
```

The equivalent smoke wrapper is:

```bash
GPU_ID=0 scripts/run_smoke_test.sh
```

If an extension fails to compile, verify `python --version`, `nvcc --version`,
and `python -c 'import torch; print(torch.__version__, torch.version.cuda)'`.
Do not mix a CUDA 12 PyTorch wheel with CUDA 11.7 extension builds.
