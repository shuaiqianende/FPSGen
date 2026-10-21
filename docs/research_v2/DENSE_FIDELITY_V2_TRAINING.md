# Dense Fidelity V2 Training Protocol and Live Status

Each architecture candidate is a fresh Stage1 initialization with 20,000
successful frame exposures / 5,000 successful updates, global batch 4 (local
batch 2 on physical GPU2+GPU3), FP16 training, AdamW, LR 1e-4 for Encoder and
2e-4 for Flow, weight decay 0.01, gradient clip 1.0, 500-update warmup,
cosine schedule, and EMA 0.999. The data manifests, NFE=2, threshold=0.5,
geometry, target generation, and Flow initialization policy remain fixed.

| Variant | Config | Training state | 900-frame EMA result |
|---|---|---|---|
| E1 Bottleneck-96 | `configs/research_v2/hier_occ_flow/fidelity_v2_e1_bottleneck96.yaml` | Complete: 5,000 updates / 20,000 exposures | IoU 0.24872, F1 0.39418, recall ceiling 0.46979 |
| E2 Asymmetric3D rank-64 | `configs/research_v2/hier_occ_flow/fidelity_v2_e2_asymmetric3d.yaml` | Complete: 5,000 updates / 20,000 exposures | IoU 0.39083, F1 0.55787, recall ceiling 0.55613 |
| E3 Lightweight Skip | `configs/research_v2/hier_occ_flow/fidelity_v2_e3_lightweight_skip.yaml` | Active; latest observed 3,449 updates / 13,796 exposures | Pending |

All candidates passed the two-update DDP smoke. E1 and E2 completed their
registered 20K runs. E3 update records currently show finite loss and
gradients. Its 5K-exposure interim evaluation is low (IoU 0.03496), but is
treated only as a health check; final selection uses the full 20K evaluation.
Its live log is
`experiments/dense_fidelity_v2/logs/e3_lightweight_skip_20k_gpu23.log` and
rank records are under `experiments/dense_fidelity_v2/training/`.

The final E2 results are in
`experiments/dense_fidelity_v2/training/e2_asymmetric3d_20k_gpu23/validation/update_005000.json`.
Its threshold-0.5 IoU is near, but below, R0's 0.40021. The fixed-K evaluation
at 78,694 / 84,963 / 94,087 remains pending; the existing per-threshold table
is not a substitute for fixed-K ranking.

After E3 finishes, run its frozen 900-frame validation and the registered
fixed-K curves for E2/E3. Only quality-qualified candidates proceed to
trained-checkpoint PyTorch FP16 and TensorRT FP16 speed comparisons. No 100K
training or Stage2 retraining has started. Do not start competing work on
GPU2/GPU3 while E3 is active.
