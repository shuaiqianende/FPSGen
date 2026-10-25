# GPU0: original Stage1 FM Flow-only matched control

2026-10-25: user additionally authorized physical GPU0. Read-only admission found a separate user's StereoGS task; nothing is stopped or reclaimed. Require14GiB free and cap this process's PyTorch allocator at12GiB. One project training task on GPU0; existing EF1/EF2 on GPU2/3 remain unchanged in their original worktree.

This experiment **S1-FM20K is a matched original-FM control, not Shortcut Models training**. The proposed Stage1 Shortcut needs a Flow-only20K continuation reference, which EF1 Encoder-only and EF2 joint adaptation cannot supply. Separate Stage1/Stage2 Shortcut implementations remain pending engineering.

## Frozen protocol

- Strict mature Stage1 EMA SHA b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2.
- Encoder fully frozen and in eval, including BN statistics. Train original Flow only. No new architecture, step embedding or consistency loss in this control.
- Same original raw Surface-hit labels,17870 Train/900 Development, geometry, original XYZ and Gaussian-to-binary FM.
- True Batch2, seed20261032;20K successful exposures /10K successful updates. AdamW Flow LR1e-5, wd.01, warmup500/cosine, clip1, EMA.999, FP16 AMP.
- The future Stage1 Shortcut must match this source, frozen Encoder, LR/schedule, source-noise stream and data order. Use a separate RNG for step-size/bootstrap sampling. Same exposure budget does not imply same teacher compute cost.
- First32/100/500 successful updates are health checkpoints within the same formal20K budget; no optimizer/scheduler reset and no discarded32-update fine-tuning phase.

## Engineering and resume scope

28 CPU tests pass, covering Flow gradients, frozen Encoder/EMA, admission and existing adaptation guards. Real GPU engineering selects the two largest registered complete-census scenes (no truncation), runs a complete AMP forward/backward/update, checks finite positive Flow gradients, no Encoder gradients, exact frozen state, actual Flow changes and peak memory.

Checkpoint serialization restores model, EMA, optimizer, scaler, scheduler and all RNG state exactly. A separate test replays an **identical stored unscaled/clipped gradient** from the restored snapshot and requires exact resulting optimizer/EMA/model/RNG state. This tests restoration independently of native CUDA backward reductions. **It does not certify bitwise end-to-end CUDA backward/update reproducibility.** That scope is recorded in the engineering report and formal checkpoint identity. No previous failed EF report is reused or relabeled PASS; no EF numerical exception flag is used.

The original full data audit must already be PASS. Every consumed shard is rehashed before first use, and cache invalidation handles changed or recently written files. Geometry, manifest, index and original input checks remain.

Only a passing core/serialization engineering report allows formal training. OOM, zero/nonfinite gradients, frozen-state changes or checkpoint-state mismatches stop the arm without shrinking scenes, Batch or Tokens. Formal training retains original finite-gradient checks, AMP skip accounting, failure streak limit and successful-exposure counts.

## Automatic continuation

After successful10K updates, run the same900 Development evaluation with frozen mature Stage2. Report Teacher-Support and deployable predicted-support outcomes, Stage1 NFE1/2/4/8, fixed-K ceilings and fine/CD/physical-IoU/JSD metrics. Shared execution is not a formal latency benchmark. No100K, second seed or Stage2 retraining is started.

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/desescene-shortcut-pilot-v1
CUDA_VISIBLE_DEVICES='' DESESCENE_SHARED_ADAPTATION=1 DESESCENE_GPU0_FLOW_CONTROL=1 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -u scripts/research_v2/hier_occ_flow/run_gpu0_flow_control.py --config configs/research_v2/hier_occ_flow/shortcut_stage1_fm_control_20k.yaml --output experiments/desescene_shortcut_pilot_v1/S1-FM20K
```

The original active EF source SHA remains7de10390fe94e98545e8b834fc08230c10ae9d0358e45732f4813c77d713e557. This new branch/worktree owns all GPU0 code and artifacts. No vendor source is modified.
