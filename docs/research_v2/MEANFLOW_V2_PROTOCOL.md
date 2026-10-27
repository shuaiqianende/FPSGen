> Latest decision2026-10-27: traditionalFM retained; A has completed20K/41 frames; B explicitly saved/stopped by user at6664 exposures, no automatic resume. See [decision](MEANFLOW_V2_FINAL_DECISION_20261027.md). Earlier snapshots below remain historical.

> Historical launch snapshot below. Latest A15K interim evaluation: [report](MEANFLOW_V2_A_INTERIM_15K_RESULTS.md). B was reassigned to GPU3 in branch `research/desescene-meanflow-v2-b-gpu3`; this evaluator does not launch training.

# DeSeScene MeanFlow-V2 preregistered protocol

Only two authorized Stage1 arms, independently random initialized. No Stage2 training, new network, extra seed, curriculum or warm-start. Existing worktrees and GPU processes remain unchanged.

## Identity and data

- Code foundation: `638e201` (the original explicit-FP16 Scratch20K runner), separate branch `research/desescene-meanflow-v2`.
- Read-only evidence snapshot: `d6a3a60`; five requested sources are SHA-indexed in `meanflow_v2/original_loss_audit.json`.
- Batch2, seed20261032, 10,000 successful updates /20,000 successful frame exposures per formal arm. Discarded engineering/32-update Smoke weights never initialize formal runs.
- Manifest SHA train: `8dc632d341ffc2c6f0c139322cd24767616d570cf84c91fb4e516a26d7bab064`.
- Manifest SHA Development: `6447076563b6c0efffbd3a07c5f695ebedf8aef1fa270a5cda9d99df73b30d53`.
- Original 17,870 train frames /900 development frames, seq08 absent from both. Development is used for existing data identity and maximum-scene engineering checks, not new900-frame quality evaluation.
- Geometry SHA `c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b`; original surface-hit labels, full384x384x32 grid, no crop/subsample.
- Encoder LR1e-4 /Flow LR2e-4; AdamW WD0.01; successful-update warmup500, original cosine; clip1; EMA0.999; initial scaler2048. Same 75%diagonal Bernoulli sampler and source/interval generators as the original run.
- Explicit FP16 network and compatible activations; FP32 master weights/Adam/EMA/statistics/time mathematics/source/integration/loss reduction. No global default dtype change.

## Mathematical arms

Forward time: `x_t=(1-t)*source+t*target`, `v_cond=target-source`, `delta=s-t>=0`; hold s fixed in JVP.

A: retain original native JVP `J=D_(v_cond,1,0)u(x,t,s)`, detached original target `v_cond+delta*J`, per-valid-voxel scene MSE L_i.

B: detached boundary estimate `v_est=u(x,t,t)`; native JVP `J=D_(v_est,1,0)u(x,t,s)`; `V_pred=u(x,t,s)-delta*stopgrad(J)` and L_i=`mean_valid((V_pred-v_cond)^2)`.

Both: `w_i=(stopgrad(L_i)+0.01)^(-1)` and loss=`mean_i(w_i*L_i)`. Epsilon0.01 is in **mean-valid-voxel squared-velocity units**, not the spatial error-sum units of the official implementation. The weighted scalar is bounded but the raw MSE need not be; report both.

At delta0, B raw regression is instantaneous FM; weighting remains enabled. All-diagonal batches skip derivative/extra-boundary work exactly as the original diagonal shortcut. Finite batches genuinely use a detached network boundary tangent in B. The prediction forward retains Encoder gradients; the JVP/tangent branches do not build higher-order reverse graphs. Extra target passes preserve buffers and checkpoint flags; the current Flow has no running-BN modules. No auxiliary head or CFG.

References (time conventions converted, not copied):
- [Mean Flows](https://arxiv.org/abs/2505.13447)
- [Improved Mean Flows](https://arxiv.org/abs/2512.02012)
- [Understanding, Accelerating, and Improving MeanFlow Training](https://arxiv.org/abs/2511.19065)
The last paper motivates later separate curriculum work only; no curriculum is included here. B tests the boundary-velocity /v-space formulation, not every architectural feature of the full iMF paper.

## Admission, safety and sequence

PhysicalGPU2 only, minimum22GiB free and no existing compute task. GPU3's existing Stage2 task is untouched; GPU0/1 also busy at the initial audit. No shared compute, process termination or memory reclamation.

A gate -> discarded32 successful-update Smoke -> fresh scratch formal20K ->41-frame evaluation -> interval analysis/paired comparisons -> standalone5+20 Stage1 timing. Only then B follows its own complete gate/smoke/formal/evaluation chain. An error blocks the remainder; no fallback, automatic restart or health-threshold relaxation.

The original AMP health bounds are unchanged: >=3 consecutive skips or skips>max(2,1%attempts) stops; successful budgets exclude skips. Nonfinite predictions/losses/JVP targets, disconnected trained gradients, unexplained state changes or failed original numerical replay bounds stop immediately. Original source/interval RNG, masters, optimizer/scheduler/scaler/EMA, sample cursor and successful counts are in every resume checkpoint. Code/config hashes and exact resume identity are mandatory.

## Diagnostics and evaluation

Per scene: raw MSE, AW loss/weight, t/s/delta, diagonal flag, native JVP RMS/max, conditional velocity RMS, B boundary RMS/error, delta*J RMS, positive/negative target MSE/counts. Per batch: preclip global gradient norm, clipped module norms, clip scale/indicator, AMP skips, attempts/successful counts, records/data order, elapsed time and peak memory.

Interval bins:0,(0,.125],(.125,.25],(.25,.5],(.5,.75],(.75,1]. Analyze mean/median/p95/p99/max, retained raw tails, sparsity and time dependence. Batch gradient stats must not be mislabelled as independent scene gradients. Logs preserve failed/skipped attempts.200/500-successful-update summaries are generated after each formal run.

Evaluation: exact original seq08 skip10041 frames0..4000, EMA, NFE1/2/4/8, threshold0.5. Stage1 standalone **coarse Parent centers**, all GT included. The inherited legacy validation dict retains its old `point_mode` label solely for strict compatibility; the new evaluation protocol explicitly records the actual coarse-center scope.

Record Parent P/R/F1/IoU/count, Fine-GT Ceiling; complete Fixed-K84963/120000; exact CUDA KD-tree CD; corrected physical IoU0.5/0.2/0.1; JSD-BEV/3D. Same frozen GT/input hashes and source-seed policy. Compare raw trajectories0->1 versus0->.5->1 and four/eight steps using identical noise.

Threshold0.3/.4/.5/.6/.7 sweep is **exploratory on41**; no best threshold is selected or called an independent-test improvement. A separate calibration-selected threshold is NOT TESTED in the first implementation. Do not change the primary0.5 result.

Paired contiguous5-frame-block bootstrap compares to original Scratch20K FM and MF; one sequence /one training seed, not cross-seed confidence. Stage1 timing is separately run without metrics/hooks/disk I/O,5warmup+20timed for each NFE. No Stage2/Jitter/full-system speed claim is made by this Stage1 benchmark. No900-frame evaluation or extra training is auto-launched.

## Commands

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/desescene-meanflow-v2
# One-shot ordered sequence; output must be fresh. Never duplicate an active queue.
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -u \
 scripts/research_v2/meanflow_v2/sequence.py --output experiments/desescene_meanflow_v2/sequence
# Explicit reviewed resume, same config/output/source, only if the earlier process exited:
# worker.py --config configs/research_v2/hier_occ_flow/meanflow_v2_a_gpu2.yaml --mode formal --resume <own resume_latest.pt>
```

Raw checkpoints/logs stay under ignored `experiments/desescene_meanflow_v2/`. Git includes code, configs, tests and compact reports only. Never push the public `origin`; use private remote `private`.
