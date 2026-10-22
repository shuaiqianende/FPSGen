# Dense Fidelity V2: Single-GPU 20K-Exposure Follow-up

**Snapshot:** 2026-10-22 19:44 CST
**Code branch/base:** `research/hs-occflow-dense-fidelity-v2-longrun`, launch commit `fdb4642a8f9e01755b97c72cfa5fb69f8b99ccba`
**Status:** R0 and E3 completed training and final 900-frame EMA validation; E2 and E1 are training. Their final quality is not yet measured.

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
| E2 Asymmetric3D rank-64 | 2 | 1,500/10,000 updates; 3,000/20,000 exposures | 0.13969 | Updates 1,001–1,500: 0.14763 | Not run; registered for 900 frames after training |
| E1 Bottleneck-96 | 3 | 385/10,000 updates; 770/20,000 exposures | 0.62166 | Updates 1–385: 1.18125 | Not run; registered for 900 frames after training |

E2's mean training loss by update block was 1.02025 (1–500), 0.21324 (501–1,000), and 0.14763 (1,001–1,500). These are stochastic flow-matching training losses, not quality metrics and not guaranteed to decrease monotonically. Both configs schedule a 12-frame interim evaluation at update 5,000 and a 900-frame EMA evaluation at update 10,000. The GPU3 supervisor was gated on E3's completed 900-frame evaluation and launched E1 only after that gate passed.

At the snapshot, both training processes were active. GPU2 reported 10.7 GiB free at 37% utilization; GPU3 reported 3.4 GiB free at 100% utilization. No existing process was stopped. Resource readings are transient and are not performance measurements.

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
