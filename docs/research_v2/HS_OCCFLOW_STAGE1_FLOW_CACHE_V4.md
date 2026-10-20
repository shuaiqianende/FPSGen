# HS-OccFlow Stage1 Flow Condition Cache V4

## Implementation

The Stage1 FM U-Net now has an opt-in inference context for projecting its three static VoxDet condition scales once, then reusing those projections across Euler steps. State, time embedding, stem, downsampled state, FiLM blocks, upsampling and FP32 Euler integration remain dynamic and unchanged. The training `forward` and default `sample(..., condition_cache=False)` path retain the original execution.

Enable it with:

```bash
--stage1-flow-condition-cache
```

The inference benchmark reports the condition-projection cache CUDA time separately and includes it in `stage1_flow_total_ms`; this prevents counting only the reduced per-step U-Net calls. The switch is included in the protocol JSON. The V4 runner also exposes independent encoder/Flow memory-format, cuDNN benchmark, TF32 and no-grad compile diagnostic flags, all disabled by default.

## Correctness checks so far

- CPU, two-step FP32 velocity and endpoint comparison: exact (`torch.equal`).
- Cache preparation is rejected when gradients are enabled.
- Reusing the context with a different observed mask is rejected.
- Fixed project environment: `26 passed` across the V4 cache tests and existing Stage1 runner suite, with CUDA hidden to ensure these are CPU tests.
- `py_compile` and `git diff --check` passed.
- PyTorch 2.0.1 CPU Dynamo probe: eager execution and `torch.compile(..., backend="eager")` passed under both `no_grad` and `inference_mode`. Default Inductor compilation of a small CPU Flow passed under `no_grad` (21.734 s cold compile) and failed under `inference_mode` with `RuntimeError: Inference tensors do not track version counter`, at `batch_scalar(...).reshape(-1)` in the FM time path. This corroborates the V3 failure mechanism, but is not a GPU performance result.

## GPU status

The first V4 cold-start one-frame baseline smoke succeeded on GPU2 (frame 100, 3.01 GiB peak allocation), but its 3.51 s latency is a cold-start smoke and is not a performance result. A cache smoke attempt was rejected by GPU admission after an external process appeared. The formal V3 baseline 5+20 run completed on GPU3 and is recorded in `HS_OCCFLOW_STAGE1_PROFILE_V4.md`. Subsequent checks found external compute processes on both permitted GPUs (GPU2 around 15–59% utilization; GPU3 around 30–91%), so cache GPU parity, detailed Stage1 profiling, layout/TF32/cuDNN, CUDA Graph and compiled GPU tests remain `NOT RUN` until an admission-safe window.

No speedup or geometry parity claim is made yet for the GPU cache path. Final parity requires exact Stage1 score and Candidate ID SHA matches plus exact Fine Occupancy equality on all 20 timed frames.
