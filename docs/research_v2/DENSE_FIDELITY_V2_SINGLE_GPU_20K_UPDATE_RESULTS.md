# Dense Fidelity V2: Single-GPU 20K-Exposure Follow-up

**Latest snapshot:** 2026-10-22 23:15 CST
**Code branch/base:** `research/hs-occflow-dense-fidelity-v2-longrun`, launch commit `fdb4642a8f9e01755b97c72cfa5fb69f8b99ccba`
**Status:** R0 and E3 completed training and final 900-frame EMA validation. E2 and E1 are still training; both completed their registered 5,000-update / 12-frame interim EMA evaluation. The final 900-frame evaluation for E1/E2 has not run.

This is a separate matched cohort from the earlier two-GPU, global-batch-4, 5,000-update results in [`DENSE_FIDELITY_V2_FINAL.md`](DENSE_FIDELITY_V2_FINAL.md). Do not combine the cohorts' metrics. This follow-up uses one physical GPU per run, local/global batch 2, 10,000 successful updates, and 20,000 successful frame exposures. It was run to compare R0 and E3 under this single-GPU schedule and to continue with E2/E1 candidates.

## Completed matched R0/E3 comparison

Both models were trained from scratch for 20,000 successful frame exposures with the same seed, schedule, data manifests, geometry, objective, and EMA policy. Evaluation uses the same 900 development frames, EMA weights, NFE=2, and strict score threshold `> 0.5`.

| Variant | Macro IoU | Macro F1 | Macro Precision | Macro Recall | Fine-GT recall ceiling | Mean active parents |
|---|---:|---:|---:|---:|---:|---:|
| R0 Original | 0.420339 | 0.587249 | 0.708653 | 0.512968 | 0.590896 | 99,303 |
| E3 Lightweight Skip | 0.420464 | 0.587482 | 0.711447 | 0.512229 | 0.590616 | 98,572 |
| E3 − R0 | +0.000125 | +0.000232 | +0.002793 | −0.000739 | −0.000280 | −731 |

The 900 validation frame IDs and order match exactly. E3's aggregate threshold metrics are effectively tied with R0 in this cohort. Per-frame performance still varies: E3 IoU is lower on 399/900 frames, with the largest paired drop −0.01613; its paired IoU difference ranges from −0.01613 to +0.01130. This is not evidence of strict per-frame parity.

These runs do **not** include fixed-K ranking metrics, a new encoder speed benchmark, a full Stage2 system comparison, or 41-frame/408-frame point-cloud quality evaluation. The separate prior-cohort fixed-K and TensorRT results remain in the historical report and are not attributed to these checkpoints.

## Active candidate runs at snapshot time

| Variant | GPU | Progress | Current loss | Recent block mean loss | Final validation |
|---|---:|---:|---:|---:|---|
| E2 Asymmetric3D rank-64 | 2 | 8,041/10,000 updates; 16,082/20,000 exposures | 0.075776 at latest row | Updates 7,542–8,041: 0.066728 | 900 frames not run; scheduled after training |
| E1 Bottleneck-96 | 3 | 5,704/10,000 updates; 11,408/20,000 exposures | 0.054542 at latest row | Updates 5,205–5,704: 0.072136 | 900 frames not run; scheduled after training |

Both configs schedule a 12-frame interim evaluation at update 5,000 and a 900-frame EMA evaluation at update 10,000. The GPU3 supervisor was gated on E3's completed 900-frame evaluation and launched E1 only after that gate passed. At the 23:15 log read, E1 had reached update 5,704 and E2 update 8,041; these values can advance after this snapshot. Linear extrapolation from each run's latest 500-update mean step time gives roughly 2.45 h remaining for E1 and 1.09 h for E2, before the final 900-frame evaluations; this is a rough ETA, not a scheduler guarantee.

### Matched 500-update training-loss windows

Means below are computed from finite successful-update rows in each run's `train_rank0.jsonl`. Every run uses global batch 2, so an update window has the same nominal exposure count. The final row is a partial window, limited by E1's latest common update (5,647). R0/E3 are completed runs; E1/E2 are ongoing. Loss is stochastic FM training loss, not an occupancy-quality metric. Instantaneous latest-row loss can fluctuate and should not be compared as a block average.

| Successful updates | R0 Original | E3 Lightweight Skip | E1 Bottleneck-96 | E2 Asymmetric3D-r64 |
|---|---:|---:|---:|---:|
| 1–500 | 1.009248 | 1.001313 | 1.020411 | 1.020295 |
| 501–1,000 | 0.213889 | 0.211916 | 0.217943 | 0.213239 |
| 1,001–1,500 | 0.147362 | 0.147303 | 0.147409 | 0.147633 |
| 1,501–2,000 | 0.128594 | 0.129660 | 0.127766 | 0.130382 |
| 2,001–2,500 | 0.113757 | 0.114529 | 0.109887 | 0.112861 |
| 2,501–3,000 | 0.099207 | 0.099496 | 0.094890 | 0.096698 |
| 3,001–3,500 | 0.087597 | 0.088254 | 0.085568 | 0.087114 |
| 3,501–4,000 | 0.083561 | 0.083434 | 0.081853 | 0.083251 |
| 4,001–4,500 | 0.078122 | 0.078654 | 0.077052 | 0.078887 |
| 4,501–5,000 | 0.073467 | 0.073321 | 0.072832 | 0.073548 |
| 5,001–5,500 | 0.071543 | 0.071608 | 0.070829 | 0.071892 |
| 5,501–5,704 (partial) | 0.076046 | 0.076071 | 0.075702 | 0.075495 |

For a direct latest-window comparison, updates 5,205–5,704 average 0.072725 (R0), 0.072625 (E3), 0.072136 (E1), and 0.072754 (E2). E2's later, non-overlapping updates 7,542–8,041 average 0.066728; do not compare that later-stage number directly against E1's 5,205–5,704 window.

Across the common 1,001–5,704 range, E1's window means are modestly lower in most recent windows; E2 remains close to R0/E3. The differences are small, and they do not establish a quality ranking.

### Interim 12-frame EMA evaluation at update 5,000

All four runs evaluated the exact same 12-frame set, in the same order, at NFE=2, EMA, threshold 0.5. These are health checks only; do not substitute them for the registered 900-frame final evaluation. R0/E3's final 900-frame results remain the primary completed comparison.

| Variant | Macro IoU | Macro F1 | Macro Precision | Macro Recall | Fine-GT recall ceiling | Mean active parents |
|---|---:|---:|---:|---:|---:|---:|
| R0 Original | 0.400263 | 0.568872 | 0.678215 | 0.500031 | 0.576161 | 97,521 |
| E3 Lightweight Skip | 0.397734 | 0.566628 | 0.658156 | 0.508213 | 0.584250 | 101,632 |
| E1 Bottleneck-96 | 0.316061 | 0.474661 | 0.523777 | 0.447166 | 0.524138 | 111,294 |
| E2 Asymmetric3D-r64 | 0.391942 | 0.560569 | 0.680828 | 0.486211 | 0.562640 | 94,194 |

E1's 12-frame IoU is notably below R0/E3 at this milestone, with lower precision and recall despite predicting more active parents. This is a warning signal, not a final rejection: the 12-frame subset is small and the registered comparison is at 10,000 updates over 900 frames. E2 is closer to R0/E3 but is also below R0 on this interim subset. No candidate-specific threshold adjustment was performed. Per-frame rows and threshold sweeps are retained in each `validation/update_005000.json`.

### Interim quality delta versus R0 at update 5,000

| Variant | Δ Macro IoU | Δ Macro F1 | Δ Precision | Δ Recall | Δ Fine-GT ceiling | Δ Active parents |
|---|---:|---:|---:|---:|---:|---:|
| E3 | −0.002530 | −0.002243 | −0.020059 | +0.008182 | +0.008090 | +4,112 |
| E1 | −0.084202 | −0.094211 | −0.154438 | −0.052865 | −0.052023 | +13,774 |
| E2 | −0.008321 | −0.008303 | +0.002613 | −0.013820 | −0.013521 | −3,326 |

## Existing architecture evidence and decision context

The following evidence comes from earlier, separately registered cohorts and is included for context; it must not be pooled numerically with the current single-GPU batch-2 cohort.

- The earlier global-batch-4 / 5,000-update architecture pilot found R1 Bottleneck-64 substantially worse than R0 (900-frame macro IoU 0.30716 vs 0.40021) and R2 OMD-3D worse still (0.25643). R1's fixed-K precision/recall and Fine-GT ceiling also declined, consistent with degraded ranking and excess false positives. The static R2 context scales stayed small (absolute maximum 0.00560); the available evidence does not prove which mechanism caused its regression. See [`DENSE_FIDELITY_V2_FINAL.md`](DENSE_FIDELITY_V2_FINAL.md) and [`DENSE_FIDELITY_V2_FAILURE_AND_SENSITIVITY_AUDIT.md`](DENSE_FIDELITY_V2_FAILURE_AND_SENSITIVITY_AUDIT.md).
- In that earlier cohort, E1 Bottleneck-96 scored macro IoU 0.24872 / Fine-GT ceiling 0.46979; E2 Asymmetric3D-r64 scored 0.390837 / 0.556141; E3 LightweightSkip scored 0.395728 / 0.571696; R0 scored 0.400207 / 0.573662. At fixed K=84,963, IoUs were 0.28637, 0.368421, 0.367221, and 0.371353, respectively. E1 and E2 missed the registered quality floors; E3 passed the screen but remained a tradeoff versus R0. These are different checkpoints from the current single-GPU 10,000-update runs.
- Random-initialized engineering speed screens at production input shape measured full Encoder means of R0 177.38 ms, E1 170.35 ms, E2 157.11 ms, E3 165.53 ms. These are not trained quality results. The isolated first-block screen measured E2 24.84 ms and E3 22.60 ms versus R0 30.13 ms; only this first-block number should not be interpreted as whole-Encoder speed.
- The completed current-cohort 900-frame comparison shows R0 and E3 nearly tied in macro threshold metrics: IoU 0.420339 vs 0.420464, F1 0.587249 vs 0.587482. E3 IoU was lower on 399/900 frames, with worst paired delta −0.01613, so it is not per-frame parity.
- TensorRT FP16 performance from the mature V7 model/checkpoint (about 143 ms Encoder, 331 ms full generation) is a separate deployment reference and is not a fair timing comparison for these new 20K scratch checkpoints. The current architecture study's TensorRT evaluation should use the corresponding candidate checkpoint after quality screening.

### Current architecture definitions

All candidates start from scratch and only change the Stage0 blocks in the two dense 3D backbones; they retain the LiDAR tokenizer, Voxel Neck, Local/TPV Neck, Stage1 Flow U-Net, feature-pyramid shapes, 384×384×32 Stage1 grid, 128-channel outputs, and training objective.

| Variant | Voxel and Local Stage0 change | Intended question |
|---|---|---|
| R0 | Original two full-width 3×3×3 128→128 convolutions per residual block; original learned 3×3×3 skip on the first block of each backbone | Quality/speed reference |
| E1 Bottleneck-96 | Main path 128→96 1×1×1, two 96→96 3×3×3, 96→128 1×1×1; original skip | Does a less aggressive bottleneck than prior R1-64 recover fidelity? |
| E2 Asymmetric3D-r64 | First 128→128 3×3×3 retained; second spatial convolution factorized as 128→64 1×1×1, 64→64 3×3×3, 64→128 1×1×1; no added intermediate activation/norm | Can one full-width spatial transform plus a factorized second transform retain quality? |
| E3 LightweightSkip | Full-width main path retained; first block's learned 128→128 skip changes from 3×3×3 to 1×1×1, with original skip normalization | Can reducing only skip-path compute preserve quality? |

Exact module-level details and CPU shape/gradient checks are in [`DENSE_FIDELITY_V2_BLOCKS.md`](DENSE_FIDELITY_V2_BLOCKS.md).

## Handoff: decisions after E1/E2 finish

No additional architecture run is currently registered or queued. The only active arms in this cohort are E1 and E2. Do not describe possible follow-ups below as experiments already performed.

1. Let the registered 10,000-update / 20,000-exposure E1 and E2 runs reach their planned endpoints unless an explicit stop decision is made. E1's update-5,000 result is a strong warning; E2 is closer to R0 but has not yet met a final quality gate.
2. Run the registered 900-frame EMA evaluation for each completed arm on the same development manifest, NFE=2 and strict score threshold `> 0.5`. Include macro and micro threshold metrics, per-frame worst cases, and fixed-K ranking at the already registered K values if the evaluation artifact supports it. Do not tune threshold/K after seeing candidate results.
3. Compare quality and measured Encoder latency together. The existing random-init speed screen suggests E2 has the largest engineering speed potential, but its trained-model advantage is not established. E1 should not be retained on loss alone if its final precision/recall remain weak.
4. Only if one candidate has a worthwhile quality-speed tradeoff, consider a narrowly targeted next run: compress Local Stage0 only, Voxel Stage0 only, or selected blocks; alternatively test a less aggressive E2 rank (for example rank 96) after a real module-speed screen. These are proposals, not committed experiments.
5. Do not start 100K training or Stage2 adaptation until the 20K 900-frame comparison and matched trained-model speed benchmark justify it. Any longer run should include a matched R0 control and a stated decision criterion.

### Current measurement limitation

The report snapshot uses file-level training logs and saved validation JSON on the experiment filesystem. The ordinary shell at snapshot time could not query the NVIDIA driver with `nvidia-smi`; therefore live GPU utilization and process state are not asserted here. Checkpoints, per-frame validation rows, and training JSONL remain local and are excluded from Git; the summary metrics and reproducibility metadata are recorded in this report and the linked protocol docs. An earlier 19:44 snapshot had GPU2 at 37% utilization and GPU3 at 100%; those transient readings are historical, not current resource evidence.

## Training and evaluation protocol

- Stage1 only; no Stage2 retraining, no NFE change, and no occupancy threshold tuning.
- 20,000 successful frame exposures, 10,000 updates, batch 2, FP16 training, AdamW, warmup-cosine schedule, EMA 0.999.
- Shared train manifest SHA256: `8dc632d341ffc2c6f0c139322cd24767616d570cf84c91fb4e516a26d7bab064`.
- Shared development manifest SHA256: `6447076563b6c0efffbd3a07c5f695ebedf8aef1fa270a5cda9d99df73b30d53`.
- Geometry SHA256: `c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b`.
- Seed: `20261021`; validation seed: `20261024`; EMA; NFE=2; threshold 0.5.
- `stage1_training.py` source hash recorded at launch: `727b64f8fd4d69e4209f29fa9ae3a563428d11637409a556ba86364f9af43382` for R0/E3 and `40f762b59258ac1a2c405b5dafba18ba931351baccd4af0fe13e4f543c6f4074` for E1/E2.

| Run | Variant/GPU | Config SHA256 | Final checkpoint SHA256 |
|---|---|---|---|
| R0 | Original / GPU2 | `ac3345f3e1ac11de5447114740d52c8e9b3ed3f34fe633b4c1807e889538612a` | `e4c97c94b7c72e67ac412a3613f28cb2235c11ac4a272840970b30fc01c921b2` |
| E3 | Lightweight Skip / GPU3 | `7a2df52189bdac3ced58d64250d7a50bd3eaa865c0576078e4da97bca55aa5eb` | `210c4ba3e262aa214bc451e49bfe43ea9220d68be2d6f26d636a342346e4e983` |
| E2 | Asymmetric3D rank-64 / GPU2 | `df1d9959cd3ef61b6736c2c3e9c242bf544ced2c2d6212d82eaed156da750dae` | Pending |
| E1 | Bottleneck-96 / GPU3 | `ab7eb61be3cc48423ddd832f0e2207de92c9d1344acf0bf78e6e23e4403367a4` | Pending |

The final R0/E3 checkpoints and raw per-frame validation JSON, training JSONL logs, and E1/E2 checkpoints are retained on the experiment host and excluded from this Git commit. Their summary metrics and hashes are recorded above. Machine-readable source artifacts are under `experiments/dense_fidelity_v2_single_gpu/` locally.

## Result files

- R0 final validation: `experiments/dense_fidelity_v2_single_gpu/r0_gpu2_20k/validation/update_010000.json`
- E3 final validation: `experiments/dense_fidelity_v2_single_gpu/e3_gpu3_20k/validation/update_010000.json`
- R0/E3 training summaries and logs: the corresponding local run folders above.
- E2/E1 live progress: `train_rank0.jsonl`, `summary.json`, and `validation/` under their run folders when present.
