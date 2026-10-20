# HS-OccFlow Actual Precision Audit

## Runtime evidence

The audit ran the real frame 100 path with leaf-module forward hooks in a
diagnostic-only run. Hook overhead is substantial (about 3.8 s for this one
frame), so none of its timing is used as performance evidence. The machine
readable module report is
`experiments/hs_inference_opt_v3/profile/dtype_audit_original_frame100_v2/protocol.json`.

Observed Stage1 condition encoder output pyramid dtypes were:

```text
level 0: float32
level 1: float16
level 2: float16
```

The encoder executes under CUDA FP16 autocast, but it is mixed in practice:
input/tokenization paths and some normalization outputs remain FP32, while
most Linear/Conv3d and intermediate BatchNorm/activation outputs are FP16.
GroupNorm returns FP32 in several paths. Stage1's Flow U-Net also runs under
the existing autocast scope; Conv3d outputs are commonly FP16, GroupNorm
outputs FP32, and its final output is FP32. Therefore “Stage1 is FP16” is too
coarse a description; its actual protocol is mixed autocast.

Stage2's original PTv3 path forces fused token features to FP32 and disables
autocast around PTv3 embedding, encoder and decoder. Runtime hooks saw FP32
features at Linear, normalization and SpConv boundaries. The FlashAttention
implementation explicitly casts packed QKV to FP16 for the flash kernel and
casts its output back to the QKV dtype. Geometry and sparse indices use integer
or FP32 coordinates. Thus the primary PTv3 path is FP32 with an internal
FP16 FlashAttention kernel, not whole-network FP16.

## V3 precision variants

`amp_fp16` enables autocast around the PTv3 backbone while keeping weights and
the Euler state in FP32. `selective_fp16` additionally uses FP32 moments for
custom Point Instance Norm, returning normalized activations to the original
activation dtype. Both completed GPU inference and 41-frame quality checks.
See `HS_OCCFLOW_PTV3_MIXED_PRECISION_RESULTS.md` for paired latency and quality.
