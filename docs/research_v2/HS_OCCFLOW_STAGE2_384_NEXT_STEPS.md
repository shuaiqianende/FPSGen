# Stage2 384 remaining gates and reproducible entrypoints

For the current measured training state, completed evaluations and unresolved
quality issues, start with [the latest discussion/status report](HS_OCCFLOW_STAGE2_LATEST_STATUS.md).
The dated execution states below preserve historical decisions and queues.

## Latest user decision: direct100K successful-frame training

On2026-10-18 the user explicitly requested direct training on100K samples.
The active protocol now uses the complete17,870-frame train split, pureFM,
Oracle parent support and true GB4 DDP on GPU2/3. The former quality gate is
preserved as failed evidence and changed to monitoring by this explicit user
decision; engineering/numerical/checkpoint gates remain mandatory.

See [the actual budget and launch report](HS_OCCFLOW_STAGE2_384_FM_100KFRAMES.md).
The572-update diagnostic was safely paused at1808 successful exposures. The
remaining98192 exposures require24548 updates, ending at absolute update25120.
tmux is `fpsgen_hs_stage2_384_fm_gb4_gpu23`, window `fm100kframes`.
Output is `experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes`.

Next research work is to monitor the actual NFE quality and complete a separate
Predicted Parent calibration/full-scene audit before deployment claims. No BCE,
PointFlow or Teacher job is queued. Only count100K complete when the saved
successful-frame counter reaches100000. The older instructions below are
historical; their approval-service and paired-pilot status is superseded.

## Historical gated pilot plan

The latest user-selected mainline is **FM only, DDP GPU2/3, packed2 scenes/rank,
global batch4**. The previous paired BCE/FM queue below is historical and no
longer runs. See [the GB4 continuation](HS_OCCFLOW_STAGE2_384_GB4.md).

Code lives in the independent worktree
`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2`.
Stage1's source, checkpoints and running process were preserved.

1. Restore the automatic approval service by refilling workspace credits.
   Do not bypass failed approval through another host/GPU execution route.
2. Wait until GPU2/3 have no existing compute tasks. The launcher and benchmark
   reject busy GPUs; they never kill tasks or reclaim occupied memory.
3. Prefer freezing the actual final100K-frame Stage1 checkpoint once available.
   Register its SHA and new config/cache identity. The27K snapshot is valid for
   interface development, not final Stage2 training.
4. Execute full-scene GPU2 and GPU3 benchmarks using the actual full-training
   token audit, then build the engineering gate from both real reports.
5. Execute32-update DDP smoke and continuous21 versus20+resume1 comparison.
   Rebuild the gate with the real comparison report.
6. Run12-frame Oracle BCE/FM overfit. Check saved learning evidence for each;
   rebuild the gate only from passing artifacts.
7. Run matched full-split1K BCE/FM pilots. Publish actual quality/resource
   results before considering5K. Keep Oracle and Predicted results distinct.
8. Audit frozen Stage1 candidates on development. Measure full-training
   predicted token quantiles and maximum with the chosen policy; four sampled
   scenes are not proof of a full-training predicted maximum. Freeze threshold,
   expansion and budget from development, with recall ceilings reported.
9. Run gated5K Oracle/Predicted pilots only if learning and resources pass.
   No100K Stage2 launcher has been automatically queued.
10. Commit new code/tests/small audit reports and push only
    `private research/hs-occflow-384-stage2`. `origin` is public and must not be
    used. Git commit/push have not executed while approval is unavailable.

Example resource benchmark (not yet executed):

```bash
export HS_STAGE2_PYTHON=/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python
CUDA_VISIBLE_DEVICES=2 "$HS_STAGE2_PYTHON" scripts/research_v2/hier_occ_flow/benchmark_stage2_384.py \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_fm.yaml \
  --token-audit experiments/hier_occ_flow/stage2_384_preflight/token_audit.json \
  --physical-gpu 2 --execute \
  --output experiments/hier_occ_flow/stage2_384_preflight/gpu2.json
# Repeat on physical GPU3, then aggregate only the actual reports.
"$HS_STAGE2_PYTHON" scripts/research_v2/hier_occ_flow/check_stage2_384_evidence.py gate \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_fm.yaml \
  --gpu-reports experiments/hier_occ_flow/stage2_384_preflight/gpu2.json \
                experiments/hier_occ_flow/stage2_384_preflight/gpu3.json \
  --output experiments/hier_occ_flow/stage2_384_preflight/engineering_gate.json
bash scripts/research_v2/hier_occ_flow/smoke_stage2_384_resume.sh \
  experiments/hier_occ_flow/stage2_384_ddp_smoke
```

These commands depend on the locally stored full sparse data and frozen
Stage1 checkpoint. GitHub intentionally excludes checkpoints, environments,
feature caches, full per-frame audit tables and generated point clouds.
Run source hashes and all config SHAs are saved so an uncommitted development
tree is not misrepresented by its parent Git commit alone.

## Current execution update

Stage1 has finished and the approval service is available. Steps1–5 above are
now historical prerequisites: final Stage1 was frozen; both GPUs passed full
scene tests; real DDP restore passed. Current controlled queue tmux is
fpsgen_hs_stage2_384_pilots_gpu23. It completes32 DDp updates, runs independent
200-update Oracle BCE/FM overfits, and advances to full-split1K pilots only
when saved learning evidence passes. It does not launch100K automatically.

Attach: tmux attach -t fpsgen_hs_stage2_384_pilots_gpu23
Log: experiments/hier_occ_flow/stage2_384_preflight/pipeline.log
The active frozen checkpoint is the final100K-frame Stage1, not27K.
