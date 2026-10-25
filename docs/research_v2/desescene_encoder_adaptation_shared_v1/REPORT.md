# L0/L1 completed quality and direct EF continuation

2026-10-25. Both L0/L1 reached 10,000 successful updates / 20,000 exposures with 5 AMP skips. All 900 evaluation frames and 64 mechanism frames completed. Original full Train+Development content audit PASS. Historical sources/checkpoints are retained.

| Arm | Fine IoU % | F1 % | Precision % | Recall % | CD m | IoU0.5 % | IoU0.2 % | IoU0.1 % | JSD BEV | JSD3D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 31.745012 | 47.765668 | 64.633048 | 38.671231 | 0.217099368 | 51.816969 | 38.501513 | 14.841591 | 0.322131032 | 0.397426814 |
| L0 | 31.759346 | 47.783718 | 64.617770 | 38.693405 | 0.217203128 | 51.820988 | 38.514658 | 14.848130 | 0.322175252 | 0.397407299 |
| L1 | 31.705652 | 47.721011 | 64.722476 | 38.579369 | 0.217775484 | 51.768402 | 38.459087 | 14.824180 | 0.322661339 | 0.397608217 |

All 900 frame/seed/input/Parent-ID/raw-GT identities match B0; outputs are native fine voxel centers. All arms share mean 104,667.733 Parents and Fine-GT ceiling 0.640357409. L0 Fine IoU delta is +0.0143348 percentage points; L1 is -0.0393600 points. Relative CD changes are +0.04779% and +0.31143%. Neither meets a meaningful architecture improvement threshold. NO-GO for further expansion.

Paired CI resamples sequence/50-frame blocks, 4000 replicates; this describes a single trained-seed cohort, not cross-seed stability. Companion JSON includes worst frames and worst-decile changes.

## GPU continuation authorized by the user

- GPU2 EF1: train mature EMA Encoder only. Frozen Flow remains differentiable to its inputs.
- GPU3 EF2: jointly train the same mature EMA Encoder and Flow.
- Original FM; true Batch2; seed 20261032; same data order; 20K successful exposures / 10K successful updates per formal arm.
- Preserve maximum real Batch2, gradient, freeze, Resume and fresh 32-update smoke gates. Formal health checkpoints at 100, 500, 10000 updates resume the same optimizer/schedule.
- Shared mode is explicit opt-in: require 14GiB free at admission and cap this process PyTorch allocator at 12GiB, retaining 2GiB reserve. No preemption or token/scene/batch reduction. A failed maximum-pair check stops that arm.
- Full original data verification remains enabled. Exclusive timing is DEFERRED; shared throughput is not latency evidence. Independent branch-activity audit remains pending.
- Coordinator starts only EF1/EF2 and their 900-frame evaluation plus EF0 reference. P2 follows completed P1 review. No P3, extra seed or 100K training is scheduled.

## Reproduction

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/desescene-encoder-adaptation-shared-v1
CUDA_VISIBLE_DEVICES='' DESESCENE_SHARED_ADAPTATION=1 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python scripts/research_v2/hier_occ_flow/run_encoder_adaptation_shared.py
```

Old idle CPU coordinator 817878 had no children and no CUDA context at handoff. Only that coordinator was replaced, using the same legacy lock. No scientific or third-party process was stopped. Original registration remains in the old worktree; new source identity requires fresh engineering and smoke artifacts.
