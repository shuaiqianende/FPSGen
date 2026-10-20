# HS-OccFlow Stage1 Layout / cuDNN / TF32 V4

All experiments used the same RTX 3090 physical GPU2, fixed V3 checkpoints and inference flags. Timing runs use five real warmup frames and the same 20 timed frames. They are independent single-run screening tests; only the Flow compile candidate received three timing rounds.

| Variant | Encoder ms | Flow ×2 ms | Stage2 ×4 ms | Full generation ms | Peak allocated GiB | Decision |
|---|---:|---:|---:|---:|---:|---|
| V3 eager reference | 203.59 | 107.92 | 143.05 | 478.07 | 3.011 | baseline |
| Encoder `channels_last_3d` | 294.24 | 107.45 | 142.95 | 569.12 | 3.152 | reject: +91.05 ms end-to-end |
| Flow `channels_last_3d` | 204.33 | 99.66 | 144.28 | 474.01 | 3.011 | no default: only −4.06 ms end-to-end, changed scores/candidates/occupancy |
| `torch.backends.cudnn.benchmark=True` | 211.03 | 108.76 | 144.22 | 488.19 | 3.011 | reject: slower and numerical change |
| Stage1 TF32 flags | 204.47 | 108.46 | 144.24 | 482.80 | 3.011 | reject: slower and numerical change |

The Encoder is already under the existing FP16 autocast policy. Profiling of the original path found dense Conv3d leaves at about `137.13 ms` of the roughly `203.74 ms` Encoder total; `torch.unique` and `scatter_max` together were under `0.7 ms`. The tested Encoder channels-last layout made the Encoder substantially slower, so the tokenizer is not a justified kernel-optimization target based on this profile.

Flow channels-last received a 41-frame quality comparison against the same V3 baseline. Macro deltas were CD `+0.000067 m`, IoU@0.5 `+0.0000009`, IoU@0.2 `+0.0000177`, IoU@0.1 `+0.0000096`, JSD-BEV `−0.0000034` and JSD-3D `−0.0000094`; quality was essentially unchanged. However, its timing gain was smaller than expected variation and it is not a worthwhile default.

cuDNN benchmark and TF32 were screened with 5+20 latency and per-frame score/candidate/occupancy diagnostics. Both were slower than the matched baseline; neither received a 41-frame metric run because each failed the first performance screen. Their score and candidate hashes changed, so they are numerical variants despite having no speed value.

Raw data:

- `experiments/hs_inference_opt_v4/encoder_channels_last/gpu2_5x20/`
- `experiments/hs_inference_opt_v4/flow_channels_last/gpu2_5x20/`
- `experiments/hs_inference_opt_v4/flow_channels_last_gpu2_41/`
- `experiments/hs_inference_opt_v4/cudnn_benchmark/gpu2_5x20_numeric/`
- `experiments/hs_inference_opt_v4/tf32/gpu2_5x20_numeric/`

The cuDNN benchmark run was first rejected by the exact-parity mode at frame 100; it was then correctly rerun as a numerical-variant timing test. The reported value above comes from that complete run, not the aborted attempt.
