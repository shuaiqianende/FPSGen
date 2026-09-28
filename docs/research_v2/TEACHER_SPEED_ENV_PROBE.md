# Teacher isolated PyTorch-2 speed-environment probe

## Motivation and isolation

The formal DCD Teacher remains FP32 in `M2024-HWZ-CasFusionNet` (PyTorch 1.13 /
CUDA 11.7 / ME 0.5.4). This study creates `env/venv_pt20_cu117` to test
PyTorch 2.0.1, AMP and Inductor on GPU1 only. It does not modify formal code,
checkpoints, or the GPU2/3 DCD run.

## Compatibility matrix and protocol

Rebuild ME 0.5.4 and Chamfer3D from source. Gate FP32 ME for 100 iterations,
then FP16/BF16 independently while coordinates remain FP32. Benchmark a cached
real 180k-point Teacher batch with three warmup and ten timed steps. Compare legacy
FP32, Torch2 eager, AMP, empty-cache ablation, and finally compile pieces.

## Results and decision

### Verified environment

- Local virtualenv: `env/venv_pt20_cu117` (Python 3.9, PyTorch 2.0.1+cu117).
- MinkowskiEngine 0.5.4: rebuilt successfully for RTX 3090 (`sm_86`). The
  local, documented `spmm.cu` include-order patch in `env/patches/` is required
  by PyTorch 2.0.1; it does not alter ME computation.
- Chamfer3D: rebuilt from the FPSGen source tree and passed CUDA
  forward/backward after `torch` was imported.
- ME: FP32, FP16 autocast plus GradScaler, and BF16 autocast all passed 100
  sparse Conv/BN/ReLU forward/backward iterations. Input coordinate construction
  remained FP32 (ME internally converts sparse keys to integer coordinates).

### Real DCD Teacher benchmark

Protocol: physical GPU1 RTX 3090; DCD-only Teacher; batch=2; 180k points;
3 warmup and 10 timed steps; same real SemanticKITTI batch and source seed per
run. P0, TensorField construction, endpoint arithmetic and DCD stayed FP32.

| ID | Environment / backend | Mean ms ↓ | Peak allocated MiB ↓ | Speedup vs legacy | Finite |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | Torch 1.13 FP32, current cache calls | 2077.4 | 13109.2 | 1.000x | yes |
| P0 | Torch 2.0 FP32, current cache calls | 2101.3 | 13129.0 | 0.989x | yes |
| P1 | Torch 2.0 FP16 AMP | 1999.9 | 11871.9 | 1.039x | yes |
| P2 | Torch 2.0 BF16 AMP | 2003.8 | 11871.9 | 1.037x | yes |
| E1 | Torch 2.0 FP32, hot-path `empty_cache` disabled | 1801.0 | 13128.4 | 1.153x | yes |
| C1 | Torch 2.0 BF16 AMP, hot-path `empty_cache` disabled | **1759.6** | **11876.1** | **1.181x** | yes |

All AMP runs completed 13 steps without NaN, Inf or OOM. The loss-mean
relative difference from B0 remained below 0.6%; this is a compatibility
probe, not a training-equivalence result.

### Inductor finding

`torch.compile(..., backend="inductor")` passes for a dense MLP. The same
probe wrapped around a real ME Conv/BN/ReLU callable returns an output but
Dynamo reports three graphs, two graph breaks and **zero captured ops**. The
primary breaks are ME's autograd context assignment and SparseTensor coordinate
map-key control flow. Therefore full sparse Teacher compilation is not an
effective target in this stack; no `torch.compile` production patch is
recommended.

### Recommendation

The isolated stack is viable for further engineering work: use BF16 only around
the ME backbone and retain FP32 for coordinates, P0, endpoint and DCD. The
largest verified speed source is removing repeated hot-path `empty_cache()`;
the BF16 plus no-cache candidate is 18.1% faster than legacy FP32 and uses 9.4%
less allocated memory. Do not change the ongoing formal FP32 DCD experiment.
Promote this only in a separate speed branch after a longer numerical-parity
run; do not adopt whole-Teacher Inductor.
