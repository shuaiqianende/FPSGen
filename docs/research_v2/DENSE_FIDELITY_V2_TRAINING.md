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
| E3 Lightweight Skip | `configs/research_v2/hier_occ_flow/fidelity_v2_e3_lightweight_skip.yaml` | Complete: 5,000 updates / 20,000 exposures | IoU 0.39573, F1 0.56333, recall ceiling 0.57168 |

All candidates passed the two-update DDP smoke and completed their registered
20K runs. E3's final 900-frame EMA evaluation is recorded at
`experiments/dense_fidelity_v2/training/e3_lightweight_skip_20k_gpu23/validation/update_005000.json`.
Its 5K-exposure interim evaluation was treated only as a health check; final
selection uses the full 20K evaluation. E3's final macro IoU is 0.39573, F1
0.56333, and Fine-GT recall ceiling 0.57168.

The final E2 results are in
`experiments/dense_fidelity_v2/training/e2_asymmetric3d_20k_gpu23/validation/update_005000.json`.
Its threshold-0.5 IoU is near, but below, R0's 0.40021. Fixed-K evaluation at
78,694 / 84,963 / 94,087 is currently running for E2 and E3; the existing
per-threshold table is not a substitute for fixed-K ranking.

The registered E2/E3 fixed-K evaluation is complete at budgets 78,694,
84,963, and 94,087. E2 misses the registered threshold-IoU and recall-ceiling
gates; E3 passes all three screening floors. E3 and R0 were also measured in
three paired PyTorch explicit-FP16 timing rounds and built/run through
TensorRT FP16 Local Backbone engines on the same GPU2. E3's TensorRT path
passed the same 900-frame threshold/fixed-K quality evaluation against its
PyTorch explicit-FP16 control. Full results are in
`DENSE_FIDELITY_V2_FINAL.md` and `DENSE_FIDELITY_V2_TENSORRT.md`.

No 100K training, Stage2 retraining, or final architecture replacement has
started. E3 is a candidate for a matched longer-training study, not a
production replacement claim.
