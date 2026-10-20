# HS-OccFlow V5 Stage1 Flow CUDA Graph

Captured and replayed the fixed two-step Stage1 Flow on real Seq08 frame 100, GPU2, with the production FP16 autocast and FP32 Euler integration. The graph uses preallocated fixed timestep values `[0, 0.5]`; creating a new CUDA scalar during capture caused earlier attempts to fail. The validated inference-only Flow method checks its fixed preconditions outside capture while ordinary `forward()` retains the original validation. CPU tests: 2 passed.

| Mode | Capture time | Replay CUDA Event / wall mean | Output vs eager | Peak allocated / reserved |
|---|---:|---:|---|---:|
| Eager Flow graph | 166.39 ms | 106.46 ms | exact | 3.00 / 4.12 GiB |
| Compiled Flow graph | 160.06 ms | 59.95 ms | exact | 3.00 / 3.84 GiB |

The compiled-graph result is a single-frame smoke, not a full-generation benchmark. The same-GPU standard V4 5+20 baseline measured Stage1 Flow ×2 at 60.62 ms; the graph replay is only about 0.67 ms lower in this comparison. This is too small to establish a meaningful gain, while graph capture adds complexity and per-shape state. Accordingly, CUDA Graph is **not selected** for the default path. No claim is made about full-model speedup or multi-frame robustness.

Raw outputs: `experiments/hs_inference_opt_v5/cuda_graph/compiled_flow_frame100.json` and `eager_flow_frame100_static_times.json`.
