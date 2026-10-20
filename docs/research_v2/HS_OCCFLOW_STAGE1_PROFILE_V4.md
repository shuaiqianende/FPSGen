# HS-OccFlow Stage1 V4 Profile

Status: Stage1 profiling and condition-cache implementation completed. Additional GPU2 layout/cuDNN/TF32 and Flow compile runs completed. GPU3 was busy during these runs, so all new GPU tests used the available GPU2 and no existing task was interrupted.

## Frozen system

- V4 branch starts from V3 commit `f51a74096ce9bd4575164fce04e8f54c6fa540c3`.
- Stage1 EMA SHA: `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`.
- Stage2 Online SHA: `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`.
- PyTorch `2.0.1+cu117`; RTX 3090, batch 1, Stage1 2 NFE and Stage2 4 NFE.
- V3 features remain enabled for comparisons: GPU candidate selection, observed-child mask and occupancy decode, Stage2 static condition cache, PTv3 multilevel topology cache, PTv3 AMP FP16 and batch-1 fastpath.

## Initial matched GPU3 baseline

One registered 5-warmup/20-frame run was completed with V3 features and the V4 runner before adding the extra hash audit. It is a valid same-GPU timing snapshot, but must be repeated with the cache variant under the final V4 runner before claiming a speedup.

| Metric | Mean | P95 |
|---|---:|---:|
| Stage1 Encoder CUDA Event | 204.40 ms | 205.88 ms |
| Stage1 Flow ×2 CUDA Events | 107.80 ms | 108.13 ms |
| Stage2 PTv3 ×4 CUDA Events | 144.30 ms | 178.80 ms |
| Network CUDA Events | 456.50 ms | 490.86 ms |
| Full generation Wall Time | 480.92 ms | 523.74 ms |
| Peak allocated memory | 3.01 GiB | 3.01 GiB |

Raw records: `experiments/hs_inference_opt_v4/baseline/v3_gpu3_5x20/`.

## GPU0 Stage1 profiling (20 October 2026)

The user explicitly authorized GPU0. At admission, it was an RTX 3090 with 23,751 MiB free, 0% utilization and 29 C, while the local inference API held a 346 MiB CUDA context. The benchmark recorded this low-footprint service context and did not stop it. A later attempt at a new channels-last smoke was correctly rejected when GPU0 utilization rose to 12% and another 3,008 MiB process appeared.

The original, non-cached Stage1 path was profiled on Seq08 frames 2300, 700 and 100. The observed scan had 18,000 points in these files; the Stage1 threshold selected 126,466 coarse parents in each frame.

| Module/category (leaf CUDA event sum) | Mean ms | Notes |
|---|---:|---|
| Stage1 Encoder total | 203.74 | frame range 203.54–203.99 ms |
| Encoder Conv3d leaves | 137.13 | largest measured category |
| Encoder GroupNorm leaves | 19.38 | |
| Encoder ReLU leaves | 9.66 | |
| Encoder DensePooling leaves | 5.45 | |
| Encoder DeformConv2d leaves | 2.67 | |
| `torch.unique` in tokenizer | 0.44 | not a dominant bottleneck |
| `scatter_max` in tokenizer | 0.25 | not a dominant bottleneck |
| Stage1 Flow, 2 NFE | 108.38 | original path; frame range 108.28–108.52 ms |
| Flow condition projections | 2.94 | three projections called twice per frame |
| Flow interpolation | 2.14 | two-step total |

The leaf-event categories account for most, but not all, of the Encoder interval; the remaining time includes functional ops and framework/layout overhead. They are profiler diagnostics and not additive replacements for the synchronized full-module measurements. Tokenizer unique/scatter together are below 1 ms here, so a custom tokenizer kernel is not currently justified. Dense Conv3d is the clear Stage1 Encoder target.

Raw profile: `experiments/hs_inference_opt_v4/stage1_profile/gpu0_3scenes_originalflow.json`.

## Follow-up V4 experiments on GPU2 (20 October 2026)

Three same-GPU 5+20 V3 baselines measured `478.066`, `478.288`, and `478.010 ms` mean generation wall time. Stage1 Encoder was `203.59–203.92 ms`, Stage1 Flow was `107.92–108.08 ms`, Stage2 was `143.05–143.23 ms`, and peak allocated memory was `3.011 GiB`.

| Variant | Encoder ms | Flow ×2 ms | Stage2 ×4 ms | Generation ms | Peak allocated GiB | Output status |
|---|---:|---:|---:|---:|---:|---|
| V3 baseline (3 runs) | 203.71 mean | 107.99 mean | 143.13 mean | 478.12 mean | 3.011 | reference |
| Encoder `channels_last_3d` | 294.24 | 107.45 | 142.95 | 569.12 | 3.152 | slower; candidate IDs changed |
| Flow `channels_last_3d` | 204.33 | 99.66 | 144.28 | 474.01 | 3.011 | small/noisy gain; score/candidate/occupancy changed |
| `cudnn.benchmark=True` | 211.03 | 108.76 | 144.22 | 488.19 | 3.011 | slower; score/candidate changed |
| Stage1 TF32 flags | 204.47 | 108.46 | 144.24 | 482.80 | 3.011 | slower; score/candidate changed |
| Stage1 Flow `torch.compile`, `no_grad` | 204.57 mean | 61.01 mean | 144.51 mean | 435.78 mean | 3.334 | repeatable speedup; numerical variant, 41-frame quality passed initial thresholds |

The Flow layout’s paired single-run improvement was only about `4.06 ms` end-to-end, below the observed frame-to-frame spread; its 41-frame quality was essentially unchanged (CD `+0.000067 m`, IoU deltas within `0.002 pp`). It is not selected as a default optimization. Encoder channels-last is a clear regression. cuDNN benchmarking and TF32 did not speed up the full path and changed the Stage1 score/candidate hashes, so neither is selected.

The largest opportunity from this phase is compiling the fixed-shape Stage1 Flow. See `HS_OCCFLOW_STAGE1_COMPILE_GRAPH_V4.md` for three repeated timing rounds, compile cold-start, exact output differences, and 41-frame quality metrics.

## Matched GPU0 5+20 baseline

The final V3 configuration was rerun on GPU0 with five real-frame warmups and 20 timed frames; only native center point extraction was enabled after generation (`--skip-jitter`).

| Metric | Mean | Std | P95 |
|---|---:|---:|---:|
| Stage1 Encoder | 203.73 ms | 0.58 ms | 204.55 ms |
| Stage1 Flow ×2 | 107.85 ms | 0.22 ms | 108.11 ms |
| Stage2 PTv3 ×4 | 142.80 ms | 27.00 ms | 178.79 ms |
| Network CUDA Events | 454.39 ms | 26.76 ms | 490.98 ms |
| Full generation Wall Time | 479.00 ms | 29.44 ms | 518.13 ms |
| Peak allocated / reserved | 3.01 / 3.84 GiB | — | — |

Raw records: `experiments/hs_inference_opt_v4/baseline/v3_gpu0_5x20/`.

## Stage1 condition projection cache, GPU0 matched run

The same 5+20 frames, V3 switches, weights and per-frame seeds were rerun with only `--stage1-flow-condition-cache` added. All 20 frames had exact Stage1 score hash, exact candidate ID hash and exact Fine Occupancy coordinate equality. Encoder outputs and Stage2 timing were unchanged within measurement noise.

| Metric | Baseline | Cached | Paired change |
|---|---:|---:|---:|
| Stage1 Encoder | 203.73 ms | 203.78 ms | +0.05 ms |
| Stage1 Flow total incl. cache preparation | 107.85 ms | 106.63 ms | −1.22 ms |
| Stage2 ×4 | 142.80 ms | 143.04 ms | +0.24 ms |
| Full generation | 479.00 ms | 476.57 ms | −2.43 ms |
| Peak allocated / reserved | 3.01 / 3.84 GiB | 3.01 / 3.84 GiB | unchanged |

Cache preparation costs 1.28 ms; the two-step Flow module portion fell by 2.50 ms, yielding about 1.22 ms net Flow savings. The full-generation paired improvement is small (about 0.5%) and comes from a single 20-frame comparison; repeat rounds are still needed before treating it as a stable speedup. It does not approach the 400 ms target by itself.

Raw records: `experiments/hs_inference_opt_v4/flow_condition_cache/gpu0_5x20/`.

## Profiling breakdown

`scripts/research_v2/hier_occ_inference/profile_stage1_v4.py` measures real Seq08 frames with module-level CUDA events and host submit times. It records tokenizer input feature construction, `torch.unique`, `scatter_max`, Point MLP leaves, dense encoder leaves, Flow leaves, condition projections, interpolation and concatenation. The original Flow path is now the default; `--flow-condition-cache` profiles the opt-in cached path. Leaf timings are diagnostic and should not be blindly added to full-module timings. The script applies GPU admission and never terminates other processes.

## Next result fields

The next safe run will include three representative frames (low, middle and high candidate counts), with these reported separately:

- Encoder total and per-leaf/operator CUDA time and host submission time.
- Tokenizer filtering/features, Point MLP, unique and scatter-max.
- Flow two-step total, condition projection calls, Flow blocks, interpolation and concatenation.
- Call counts and Candidate count for each scene.

No result is inferred for unmeasured operators. See `HS_OCCFLOW_STAGE1_FLOW_CACHE_V4.md` for the cache implementation and parity status.
