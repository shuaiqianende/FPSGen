# Dense Fidelity V2 GPU Benchmark Status

## Runtime and admission

The host has four RTX 3090 cards. GPU2 and GPU3 were admitted idle for the
pilot (about 24 GiB free each, no compute PIDs). The ordinary sandbox shell
does not expose `/dev/nvidia*`, but an approved host-level execution context
does. The correct pinned runtime is
`/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python` (PyTorch
2.0.1+cu117, CUDA 11.7, torch-scatter 2.1.2, MMCV 1.7.2, SpConv 2.3.6).

This environment prepends a system CUDA library directory containing cuDNN
8.4, while the PyTorch wheel expects cuDNN 8.5. Running with
`env -u LD_LIBRARY_PATH` selects the bundled cuDNN 8.5.0 and allows the model
and benchmark to run. The previous `/FPSGen/.conda-envs/fpsgen` environment
(PyTorch 1.13, missing model dependencies) is not the Stage1 runtime.

## Stage0 block benchmark

Physical GPU2, explicit FP16, batch 1, production Stage0 shape
`[1,128,192,192,16]`, two warmups and five timed samples. These are short
engineering screens, not the registered 20/100-iteration final microbench.

| Stage0 first block | BN mean (ms) | GN mean (ms) | Parameters |
|---|---:|---:|---:|
| R0 Original | 30.13 | 30.08 | 1,327,872 |
| E1 Bottleneck-96 | 28.44 | 29.29 | 965,696 |
| E2 Asymmetric rank-64 | 24.84 | 25.83 | 1,012,480 |
| E3 Lightweight Skip | 22.60 | 23.45 | 901,888 |

The regular R0 block measured 20.89 ms (BN). Peak allocated memory for these
single-block runs was 1.28–1.78 GiB. E3 is fastest in this isolated first-block
screen; E2 is fastest among E1/E2/E3 in the full Encoder screen below. Neither
result establishes trained quality.

## Full random-initialized Encoder screen

Physical GPU2, real input `/KITTI_Odometry/00/input_/000100.npy` (18,000
points), explicit-FP16 inference, two warmups and five timed repeats. It runs
the actual tokenizer, both backbones, necks, and the three output pyramids,
but uses fresh random model weights and is therefore only an engineering
speed screen.

| Variant | Encoder mean (ms) | Median (ms) | Peak allocated (GiB) | Pyramid output |
|---|---:|---:|---:|---|
| R0 Original | 177.38 | 177.11 | 1.660 | `[1,128,16,192,192]`, `[1,128,8,96,96]`, `[1,128,4,48,48]` |
| E1 Bottleneck-96 | 170.35 | 170.26 | 1.656 | same |
| E2 Asymmetric rank-64 | 157.11 | 156.98 | 1.658 | same |
| E3 Lightweight Skip | 165.53 | 165.47 | 1.659 | same |

All outputs were finite and FP16. Relative to this same-run R0 screen, E1/E2/E3
were about 4.0%/11.4%/6.7% faster. This is not a trained-model quality or
deployment result.

The two machine-readable artifacts are
`experiments/dense_fidelity_v2/gpu_benchmark/gpu2_stage0_smoke.json` and
`experiments/dense_fidelity_v2/gpu_benchmark/gpu2_full_encoder_smoke.json`.

## Distributed training smoke and current full pilots

E1, E2, and E3 each passed a two-update DDP smoke on physical GPU2+GPU3 with
local batch 2. All reached update 2, reported finite encoder/flow gradients,
and confirmed synchronized rank parameters. The data audit verified the
registered 17,870-frame train split and 900-frame development split.

The full E1/E2/E3 20K runs use independent outputs and are launched
sequentially. Current live state is recorded in
`experiments/dense_fidelity_v2/training/` and the per-variant logs. The short
benchmark JSONs do not replace the formal 20K-trained model comparison.

At the 2026-10-21 progress check, E1 had reached 438/5,000 updates (1,752 of
20,000 successful frame exposures) at approximately 2 seconds/update. E2/E3
remain queued so the two designated GPUs are not oversubscribed.

Fidelity-specific TRT tooling is available at
`export_fidelity_backbone_onnx.py`, `build_fidelity_backbone_trt.py`,
`benchmark_fidelity_trt_encoder.py`, and `fpsgen/hier_occ_flow/fidelity_trt_adapter.py`.
They compile and expose CLI help in the Stage1 environment. No TensorRT engine
has yet been built for a fidelity checkpoint; that gate follows trained-model
quality evaluation.
