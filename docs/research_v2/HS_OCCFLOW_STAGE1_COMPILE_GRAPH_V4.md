# HS-OccFlow Stage1 Flow Compile and CUDA Graph V4

## Scope

The benchmark holds the V3 model protocol fixed: Stage1 EMA and 2 NFE, Stage2 Online and 4 NFE, threshold 0.5, the same Stage2 FP16/multilevel PTv3 configuration, same frame IDs and per-frame seeds. All GPU runs below used the same physical RTX 3090 GPU2 in separate idle windows. GPU3 was occupied by an unrelated high-load process and was left alone.

## Stage1 Flow compile

The Flow is compiled with `torch.compile(flow, dynamic=False)` and uses `no_grad`, because PyTorch 2.0.1 Inductor raises `Inference tensors do not track version counter` on the `inference_mode` time-embedding path. A matched Eager + `no_grad` run confirmed that switching from `inference_mode` to `no_grad` alone does not change Stage1 score or candidate hashes on the 20 timed frames; those hashes exactly matched the ordinary V3 baseline. The compiled Flow itself changes low-level numeric results.

| Run | Eager V3 generation mean (ms) | Compiled Flow generation mean (ms) | Paired change (ms) | Eager Flow ×2 (ms) | Compiled Flow ×2 (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 478.066 | 435.326 | −42.740 | 107.917 | 60.924 |
| 2 | 478.288 | 435.442 | −42.846 | 107.960 | 61.030 |
| 3 | 478.010 | 436.576 | −41.434 | 108.084 | 61.061 |
| Mean | 478.121 | 435.781 | **−42.340 (−8.86%)** | 107.987 | 61.005 |

The corresponding network-forward means were `454.83 ms` eager versus `410.08 ms` compiled. Encoder remained about `204 ms` and Stage2 about `144.5 ms`; the speedup is attributable to the Stage1 Flow path, not a changed NFE or Stage2 configuration. Compiled mean peak allocation was `3.334 GiB` versus `3.011 GiB` eager (reserved about `3.98 GiB` versus `3.84 GiB`).

The first compiled call took `10.13 s` wall time on warmup frame 0. Warm timings then ran without continuous recompilation. Dynamo recorded 2 unique graphs, 162 captured calls and 2 graph breaks at tensor-to-bool validation. Compilation setup itself was about 2.5 ms; the cold compilation is lazy and occurs on the first forward. The initial one-frame smoke measured about 12.9 s and is not a latency result.

Runs:

- `experiments/hs_inference_opt_v4/baseline/v3_gpu2_5x20/`
- `experiments/hs_inference_opt_v4/baseline/v3_gpu2_repeat2_5x20/`
- `experiments/hs_inference_opt_v4/baseline/v3_gpu2_repeat3_5x20/`
- `experiments/hs_inference_opt_v4/compile/stage1_flow_eager_gpu2_5x20_no_grad/`
- `experiments/hs_inference_opt_v4/compile/stage1_flow_gpu2_5x20_no_grad/`
- `experiments/hs_inference_opt_v4/compile/stage1_flow_gpu2_repeat2_5x20_no_grad/`
- `experiments/hs_inference_opt_v4/compile/stage1_flow_gpu2_repeat3_5x20_no_grad/`

## Output differences and quality gate

The compiled scores, candidate IDs and Fine Occupancy are not bitwise-identical to eager. All 20 timed frames had different score and candidate hashes; the 41-frame quality prediction set had different scores/candidates on all 41 frames. Mean candidate count changed by `+37.7` parents, and Fine Occupancy symmetric difference averaged `5,797` cells (maximum `11,019`). Thus this is a numerical variant, not an exact cache-style optimization.

The 41-frame evaluation used the same raw-map GT, ROI, exact GPU KD-tree CD, corrected physical-grid IoU and JSD functions as the V3 quality reference:

| Metric | V3 Eager | Compiled Flow | Change |
|---|---:|---:|---:|
| CD (m) ↓ | 0.202997 | 0.202787 | −0.000210 |
| IoU@0.5 ↑ | 0.506869 | 0.506824 | −0.000045 (−0.00445 pp) |
| IoU@0.2 ↑ | 0.395382 | 0.395352 | −0.000030 (−0.00305 pp) |
| IoU@0.1 ↑ | 0.164187 | 0.164191 | +0.000004 (+0.00045 pp) |
| JSD-BEV ↓ | 0.307663 | 0.307588 | −0.000076 |
| JSD-3D ↓ | 0.416174 | 0.416162 | −0.000013 |
| Occupied voxels / frame | 613,845 | 613,950 | +106 |

The macro changes are within the task's initial quality screening thresholds. Per-frame maxima were: CD regression `+0.000303 m` (frame 2100), IoU@0.5 regression `−0.000786` (frame 1200), IoU@0.2 regression `−0.000255` (frame 3100), and IoU@0.1 regression `−0.000145` (frame 3100). No frame shows a large aggregate metric regression, though the cell-level occupancy is numerically different.

Quality artifacts:

- Prediction set: `experiments/hs_inference_opt_v4/quality/stage1_flow_compile_gpu2_41/`
- Metrics and per-frame CSV: `experiments/hs_inference_opt_v4/quality/stage1_flow_compile_gpu2_41/metrics.json` and `.csv`
- Timing records: `experiments/hs_inference_opt_v4/compile/stage1_flow_gpu2_5x20_no_grad/`, repeat2 and repeat3 directories above.

## CUDA Graph status

**NOT TESTED on GPU.** The Stage1 Flow has static dense shapes and fixed 2-step times, so capture may be feasible, but its ordinary `forward` includes a tensor-to-Python boolean range check (`bool(((t >= 0) & (t <= 1)).all())`) that synchronizes and is not capture-safe. A graph path would need a separately validated inference-only entry that validates the fixed schedule before capture, captures the exact two Flow calls and FP32 Euler integration, and safely refreshes state/condition buffers each frame. Since Flow compilation already saves about 42 ms/frame with acceptable 41-frame aggregate metrics, CUDA Graph engineering is deferred rather than introducing another unvalidated route in this milestone.

## Recommendation

Keep the eager V3 path as the default. Expose `--compile-target stage1_flow` only as an explicit opt-in numerical tradeoff until a larger quality set and deployment policy approve it. Do not enable it silently: candidate IDs and the final occupancy do change. CUDA Graph remains an optional follow-up; it is not required to obtain the measured speedup.
