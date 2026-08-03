# Installation

This project has CUDA extensions. Use one consistent Python, PyTorch, CUDA
toolkit, and C++ compiler from creation through training; rebuilding is needed
after changing any of them.

The validated compatibility target is Python 3.9, PyTorch 1.13.0,
`pytorch-cuda=11.7`, PyTorch Lightning 1.8.1, and MinkowskiEngine 0.5.4.
An NVIDIA driver new enough for CUDA 11.7 and a GPU with compute capability
supported by that PyTorch build are required.

```bash
conda env create -f environment.yml
conda activate fpsgen
```

Install MinkowskiEngine after PyTorch is available. It is compiled locally and
must see the same CUDA toolkit used by PyTorch:

```bash
export CUDA_HOME=/usr/local/cuda-11.7   # adapt only if your toolkit differs
export MAX_JOBS=8
pip install --no-deps MinkowskiEngine==0.5.4
```

Then install FPSGen and its two local CUDA extensions:

```bash
pip install -r requirements.txt
pip install -e .
pip install -e fpsgen/models/ChamferDistancePytorch/chamfer3D
pip install -e fpsgen/utils/metrics_gen/pytorch_structural_losses
```

The bundled Chamfer implementation originates from
`ThibaultGROUEIX/ChamferDistancePytorch`; its MIT license is retained at
`third_party/licenses/ChamferDistancePytorch-MIT.txt`.

Finally, select a physical GPU and run both checks. CUDA remaps the selected
physical device to logical `cuda:0` inside Python.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py
CUDA_VISIBLE_DEVICES=0 python scripts/verify_cuda_ops.py
```

The repository also provides the equivalent wrapper:

```bash
GPU_ID=0 scripts/run_smoke_test.sh
```

If either extension fails to compile, record `python --version`,
`nvcc --version`, `python -c 'import torch; print(torch.__version__,
torch.version.cuda)'`, and the full compiler error. Mixing a CUDA 12 PyTorch
wheel with a CUDA 11.7 extension build is unsupported.
