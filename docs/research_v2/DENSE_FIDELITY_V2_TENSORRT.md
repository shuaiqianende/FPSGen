# Dense Fidelity V2: TensorRT FP16 validation

## Result

E3 Lightweight Skip passed the registered 20K development quality gates and
was faster than the R0 Original encoder on the same RTX 3090. Replacing only
E3's Local Backbone with a static TensorRT FP16 engine reduced its full
Stage1 Encoder time further. The same engine passed the registered 900-frame
threshold and fixed-K comparison against E3's PyTorch explicit-FP16 path.

This is a Stage1 encoder result. It does not establish a full HS-OccFlow
end-to-end latency or a trained Stage2 system result: Stage2 was not retrained
for the 20K architecture pilot.

## Protocol

- Physical GPU 2, NVIDIA RTX 3090, batch size 1.
- Same 25-row development manifest for all R0/E3 encoder timing runs; first 5
  rows were warm-up and the remaining 20 were timed.
- The input manifest SHA256 was
  `224c556d7a8da18e437e53e2e67aa08cf364c909f80fe068dbb813570c7955fe` for
  both PyTorch explicit-FP16 and TensorRT FP16 timing runs.
- Static FP16 engine input shape: `[1,128,192,192,16]`; outputs are the three
  backbone pyramid levels at `[1,128,192,192,16]`, `[1,128,96,96,8]`, and
  `[1,128,48,48,4]`.
- TensorRT 10.13.3.9, PyTorch 2.0.1+cu117, CUDA 11.7 runtime for the PyTorch
  pipeline. The isolated TensorRT runtime was reused; the V6/V7 environment
  was not replaced.
- Both engines were built from the corresponding registered EMA 20K
  checkpoint. No quantization or model-parameter training was used.
- R0 checkpoint SHA256: `77ef88676eaaad0122947a4763c0e79dcab33064fd7ff2e1de73ee3d442cfe5f`;
  E3 checkpoint SHA256: `ef0daea6706fc17f6013f09b7569f3035e09edad0396f0902eb66c53f4c25142`.
- R0 engine SHA256: `7a50d12537ddcf85334e1ca7f906615ea4d975ab620eff5d9539aa77100c48da`;
  E3 engine SHA256: `112cf35f73adff0e9d0784e7040aeb1ac9b12a4832ab2ffdce8b896a14b516c9`.

## Encoder timing

| Variant / backend | Encoder CUDA event mean | Encoder wall incl. H2D mean | P95 wall | Peak allocated |
|---|---:|---:|---:|---:|
| R0 Original, PyTorch explicit FP16 | 179.78 ms | 180.14 ms | — | 1.940 GiB |
| E3 Lightweight Skip, PyTorch explicit FP16 | 167.07 ms | 167.45 ms | — | 1.939 GiB |
| R0 Original, TensorRT FP16 Local Backbone | 146.97 ms | 147.24 ms | 149.19 ms | 1.923 GiB |
| E3 Lightweight Skip, TensorRT FP16 Local Backbone | 136.81 ms | 137.09 ms | 138.19 ms | 1.923 GiB |

The paired E3/R0 PyTorch encoder difference was 12.70 ms (7.58% lower E3
latency). Under TensorRT the difference was 10.15 ms (6.90% lower E3
latency). Comparing the same E3 model across backends, TensorRT lowered full
encoder wall time by 30.35 ms (18.1%) relative to PyTorch explicit FP16.
These are encoder-only comparisons; they do not include Stage1 Flow, Stage2,
or point output.

The standalone local-backbone engine averaged 32.83 ms for E3 and 36.99 ms
for R0. Its build took 67.34 s and 67.97 s respectively. Each engine was
about 19–20 MB. These engine-only times exclude the PyTorch work before and
after the backbone and are not substitutes for full encoder timing.

## Runtime and numerical evidence

TensorRT Engine Inspector reported FP16 `CaskConvolution` layers using
half-precision convolution tactics, including the Stage0 3×3×3 128-to-128
convolutions; this was not a Fake Quant or FP16-weight-only result. The E3
engine output relative-L2 errors against PyTorch explicit FP16 were 0.00326,
0.00295, and 0.00278 for the three pyramid outputs. All outputs were finite.

The trained E3 model was evaluated on all 900 registered development samples
with the same GT manifest, NFE=2, threshold sweep, and frozen candidate
budgets. TensorRT versus PyTorch explicit-FP16 macro results were:

| Metric | PyTorch explicit FP16 | TensorRT FP16 | TRT − PyTorch |
|---|---:|---:|---:|
| Threshold 0.5 IoU | 0.39572834 | 0.39573624 | +0.00000790 |
| Threshold 0.5 active parents | 100,268.04 | 100,275.00 | +6.96 |
| Threshold 0.5 Fine-GT recall ceiling | 0.57168426 | 0.57170523 | +0.00002097 |
| Fixed K=78,694 IoU | 0.35664346 | 0.35664261 | −0.00000085 |
| Fixed K=84,963 IoU | 0.36722689 | 0.36722667 | −0.00000022 |
| Fixed K=94,087 IoU | 0.37901871 | 0.37901893 | +0.00000022 |

Across individual frames, the threshold-0.5 IoU delta ranged from
−0.0001221 to +0.0001234. The mean active-parent count difference was about
7, with a range of −71 to +67. Thus the engine changes a small number of
thresholded parents but has no material effect on the registered aggregate
or worst per-frame IoU deltas. Exact parent-ID symmetric differences were not
persisted by the curve evaluator and are not claimed here.

## Reproduction commands

Use the existing PyTorch 2.0.1 environment and add the isolated TensorRT
runtime paths for export/build/engine execution. Run from the repository
root, selecting only GPU 2 with `CUDA_VISIBLE_DEVICES=2`.

```bash
python scripts/research_v2/hier_occ_flow/export_fidelity_backbone_onnx.py \
  --config configs/research_v2/hier_occ_flow/fidelity_v2_e3_lightweight_skip.yaml \
  --checkpoint experiments/dense_fidelity_v2/training/e3_lightweight_skip_20k_gpu23/checkpoints/exposure00020000.pt \
  --module local_backbone --precision explicit_fp16 --physical-gpu 2 \
  --output-dir experiments/dense_fidelity_v2/tensorrt/e3_local_backbone_frame100
```

Build and benchmark the engine with `build_fidelity_backbone_trt.py`, then
run the in-pipeline 5-warmup/20-timed check with
`benchmark_fidelity_trt_encoder.py`. The 900-frame quality results are stored
in `experiments/dense_fidelity_v2/tensorrt/` as JSON, alongside the export
protocols and TensorRT inspector reports. Large ONNX files and engine plans
remain local and are not intended for Git.

## Decision

E3 is the only new architecture that passed all registered 20K screening
gates. TensorRT FP16 provided an additional stable encoder speed improvement
without a measurable aggregate Stage1 quality change relative to E3's
PyTorch explicit-FP16 path. E3 remains below R0's 20K Stage1 IoU by about
0.00448 absolute, so this result supports taking E3 into a controlled longer
training study; it does not justify replacing the mature production encoder
or claiming final-system quality parity. Stage2 adaptation and a matched
longer-training R0 control remain future gates.
