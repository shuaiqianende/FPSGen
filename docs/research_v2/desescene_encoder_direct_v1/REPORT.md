# User-directed formal20K numerical pilot

2026-10-25. After disclosure of the strict replay failures, the user explicitly requested: 开始正式训练啊. EF1 and EF2 now use an explicitly registered numerical-pilot exception. Original engineering reports remain FAIL. No bitwise or bounded numerical replay certification is claimed.

The only excepted checks are resume_parameter_bound and resume_auxiliary_bound in the two hash-pinned original reports. All other checks must be true: real maximum Batch2, finite positive gradients, Encoder change, frozen state, equal replay loss, RNG and scalar restoration. Observed maximum parameter differences are 1.30385e-7 / 1.34110e-7; auxiliary differences are 8.81654e-7 / 9.99239e-7. Pilot exception ceilings are 2e-7 / 2e-6 absolute and do not redefine the original strict checks. Future resumed trajectories remain unverified for numerical replay.

Strict global determinism failed at _bincount_cuda. Deterministic cuDNN alone did not pass the old replay bounds. This does not establish which individual backward kernels caused all differences; no causal claim is made.

## Formal runs

- GPU2 EF1: mature EMA Encoder only; frozen Flow retains input gradients.
- GPU3 EF2: same initial Encoder and Flow, trained jointly.
- Same registered Gaussian-to-binary FM; same original Surface-hit data; batch2; seed20261032; 10K successful updates /20K exposures per arm.
- Updates32/100/500/10000 are in-run health milestones. The first32 updates belong to the formal budget. Optimizer, EMA, schedule and data order are never reset at milestones.
- NaN/Inf, invalid/frozen gradients, frozen-state drift, three consecutive failed attempts and excessive AMP skips remain stop conditions.
- Source checkpoint hashes remain strict. Checkpoints preserve all optimizer, EMA, scaler, scheduler, RNG, data cursor and explicit exception identity.
- Full data audit PASS is hash-pinned. Original full manifest/index checks remain. Each consumed shard is verified before access; content changes invalidate cached validation. This avoids repeated full unused-data scans at startup while never feeding an unchecked shard.
- Each completed arm automatically runs the same900-frame evaluation. EF0 reference follows. Formal exclusive timing is deferred; no timing claim from shared training. P2/P3 or100K are not started here.

## Reproduction

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/desescene-encoder-adaptation-shared-v1
CUDA_VISIBLE_DEVICES='' DESESCENE_SHARED_ADAPTATION=1 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -u scripts/research_v2/hier_occ_flow/run_direct_encoder_pilot.py --policy experiments/desescene_encoder_adaptation_v1/formal_numeric_20261025_b/policy.json --output experiments/desescene_encoder_adaptation_v1/formal_numeric_20261025_b
```

The source SHA and exact evidence hashes are in policy.json. Failed historical engineering and deterministic diagnostics stay in their original folders; nothing is relabeled PASS or overwritten. This cohort is mature100K+additional20K adaptation, not scratch20K.

Startup regression: coarse filesystem timestamps can hide a same-size immediate file edit. Recently modified shards now always rehash, without changing the corruption test.31 CPU tests pass. The first startup was replaced before any model-launch audit, checkpoint or training update; its logs and policy are retained. Active path is formal_numeric_20261025_b.
