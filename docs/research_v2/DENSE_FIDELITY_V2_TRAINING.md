# Dense Fidelity V2 Training Protocol and Status

Each candidate is a fresh Stage1 initialization with 20,000 successful frame
exposures / 5,000 successful updates, global batch 4 (local batch 2 on physical
GPU2+GPU3), FP16 training, AdamW, LR 1e-4 for Encoder and 2e-4 for Flow,
weight decay 0.01, gradient clip 1.0, 500-update warmup, cosine schedule, and
EMA 0.999. The data root and train/development manifests are copied from the
registered OMD V1 pilot. NFE=2, threshold=0.5, geometry, target generation,
and Flow initialization policy remain fixed.

Three isolated configs and a dedicated launcher are present:

| Variant | Config |
|---|---|
| E1 Bottleneck-96 | `configs/research_v2/hier_occ_flow/fidelity_v2_e1_bottleneck96.yaml` |
| E2 Asymmetric 3D rank-64 | `configs/research_v2/hier_occ_flow/fidelity_v2_e2_asymmetric3d.yaml` |
| E3 Lightweight Skip | `configs/research_v2/hier_occ_flow/fidelity_v2_e3_lightweight_skip.yaml` |

Training status for all three: `NOT RUN`. GPU admission cannot pass in the
current runtime because the NVIDIA driver/device nodes are unavailable. The
runner enforces 20K exposures and GPU2+GPU3; it does not permit a smaller run to
be relabeled as the registered full pilot. The 2K/5K/10K/20K checkpoints and
900-frame validation therefore remain future work.
