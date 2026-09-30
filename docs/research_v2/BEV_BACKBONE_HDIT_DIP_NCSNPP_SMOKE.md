# HDiT-S / DiP-S / NCSNpp-S real-data smoke record

## Protocol

Environment: `/data-12/M2024-HWZ/FPSGen/env/venv_pt20_cu117` (Python 3.9, PyTorch 2.0.1, CUDA 11.7, MinkowskiEngine 0.5.4). Each run used the fixed `gt_possion` SemanticKITTI contract, `batch_size=1`, eager FP32, `runtime.compile_dense=false`, one visible RTX 3090, and `limit_train_batches=8`.

The standard run is the unmodified `fpsgen/train_bev.py` path, so it validates the existing uniform eight-condition sampler, Flow-Matching construction, weighted velocity loss, optimizer and backward path. A companion one-real-batch gradient probe then forces all conditions active only to prove condition-adapter connectivity. It does not save a checkpoint. NCSN++ uses one in-memory warmup in that probe because its retained `init_scale=0` output heads intentionally block upstream gradients at the very first backward; the normal 8-step optimizer smoke had already completed.

No row below is a quality result or a comparison/ranking.

| backbone | source / configuration | GPU | 8-step first -> last loss | gradient probe peak allocated | condition grad norm | core grad norm | result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| HDiT-S-BEV | k-diffusion `4601bf...`; `128/256/512`, `[2,2,4]` | physical GPU2 (`CUDA_VISIBLE_DEVICES=2`) | 2.91 -> 2.89 | 674.76 MB | 0.331665 | 2.208164 | PASS |
| DiP-S/16-BEV | DiP `949294...`; patch16, hidden384, groups6, blocks8 | physical GPU3 (`CUDA_VISIBLE_DEVICES=3`) | 2.49 -> 2.46 | 691.26 MB | 0.00000366 | 1.233744 | PASS |
| NCSNpp-S-BEV | score_sde `cb1f359...`; `nf=96`, 7 levels | physical GPU2 (`CUDA_VISIBLE_DEVICES=2`) | 2.30 -> 2.14 | 1716.98 MB | 0.225915 | 23.010889 | PASS |

For all three, forward output was `[1,3,256,256]`; all recorded losses and condition/core gradients were finite; no OOM or Inf/NaN occurred. The condition gradient is nonzero in every case. The DiP magnitude is small but nonzero and finite; it is reported as evidence of connectivity only, not as a quality or optimization conclusion.

## CPU/static validation

The isolated PyTorch2 environment ran `python -m py_compile` on all new modules and `python -m pytest -q tests/test_bev_backbones.py tests/test_bev_backbone_architecture.py` with **19 passed**. Guards cover: exact-zero condition pyramids/tokens, outputs and finite behavior at `t=0,0.5,1`, HDiT hierarchy and token merge/split, DiP head dimension/Detailer, NCSN++ hierarchy/FIR/skip paths, legacy lazy import, and absence of VAE runtime dependencies.

## Parameter accounting

| backbone | generator core | condition adapter | shared PointPillar | total |
| --- | ---: | ---: | ---: | ---: |
| HDiT-S-BEV | 25,456,389 | 757,760 | 1,600 | 26,215,754 |
| DiP-S/16-BEV | 39,038,467 | 3,489,792 | 1,600 | 42,529,861 |
| NCSNpp-S-BEV | 22,808,949 | 1,642,176 | 1,600 | 24,452,733 |

All generator-plus-adapter totals are within the requested 20M–60M range.

## Scope boundary

Formal training: **NOT STARTED**. Inductor: **NOT RUN**. AMP: **NOT RUN**. Quality evaluation: **NOT RUN**.
