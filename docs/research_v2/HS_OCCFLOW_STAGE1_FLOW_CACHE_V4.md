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

The first V4 cold-start one-frame baseline smoke on GPU2 (frame 100, 3.01 GiB peak allocation) is not a performance result. Later GPU2 idle windows were used for the further V4 tests documented below. GPU3 remained busy with an unrelated LiDiff process (around 7.5 GiB and 91–98% utilization in the latest checks); it was not interrupted.

## GPU0 measured result

On 20 October 2026, V3 Baseline and condition-cache variants were run sequentially on GPU0, each with the registered 5 warmup + 20 timed Seq08 frames, identical checkpoints, Stage1/2 NFE, candidate settings and per-frame seeds. At admission GPU0 had 23,751 MiB free and 0% utilization; a pre-existing 346 MiB local inference API context was idle. It was explicitly recorded and never stopped.

| Metric | Baseline | Cached | Paired delta (cached − baseline) |
|---|---:|---:|---:|
| Two-step Flow total incl. context setup | 107.85 ms | 106.63 ms | −1.22 ms |
| Two-step Flow network calls only | 107.85 ms | 105.36 ms | −2.50 ms |
| Fixed condition projection setup | 0 ms (repeated inside each step) | 1.28 ms (once) | +1.28 ms |
| Full generation Wall Time | 479.00 ms | 476.57 ms | −2.43 ms |
| Fine Occupancy parity | reference | 20/20 exact | PASS |
| Stage1 score hash / candidate ID hash | reference | 20/20 exact | PASS |
| Peak allocated / reserved | 3.01 / 3.84 GiB | 3.01 / 3.84 GiB | unchanged |

The per-frame paired wall-time difference averaged −2.43 ms (paired SD 2.31 ms) in this one round. This is a small, promising engineering gain, but repeat rounds are needed before claiming a stable speedup. It does not materially change the 400 ms optimization target.

Raw runs:

- `experiments/hs_inference_opt_v4/baseline/v3_gpu0_5x20/`
- `experiments/hs_inference_opt_v4/flow_condition_cache/gpu0_5x20/`

CPU test suite: 28 passed after adding the opt-in Stage1 memory-layout helper tests. Detailed GPU Encoder profiling also found the fixed condition projections total about 2.94 ms when recomputed twice; the cache setup costs 1.28 ms, which explains the measured ~1.2 ms net Flow reduction.


## GPU2 repeat check for the condition cache

The cache was tested on GPU2 with the V3 configuration. One 5+20 run completed with exact score, candidate-ID and Fine Occupancy hashes and mean generation `503.57 ms`, but it had substantial Stage2 timing outliers and is not a valid speed comparison. A second 5+20 attempt failed Fine Occupancy parity on frame 100 after Stage1 score and candidate parity had passed. Separate baseline and cache one-frame smokes each matched. This means exact parity across repeated full runs is **not established**; Stage2's sparse backend also has run-to-run variability. The cache remains opt-in.

The GPU0 matched result remains a single-round engineering measurement: about `1.22 ms` net Flow saving after `1.28 ms` static projection preparation, and `2.43 ms` full-generation saving. Three-round cache timing and robust repeated exact parity have not been established. Do not describe this as a major speedup.
