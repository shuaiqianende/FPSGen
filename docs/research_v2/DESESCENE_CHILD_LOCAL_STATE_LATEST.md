# DeSeScene: latest experiment evidence and next work

Updated2026-10-24. This index links completed results separately from the current unfinished pilots.

- [Current L0/L1 progress,200-step mean losses and actual continuation](desescene_child_local_progress_20261024_2140/README.md). Both arms are training toward20K successful exposures, Batch2, seed20261032, onGPU2/GPU3. Final geometry quality is not yet tested.
- [Local XYZ/attention implementation and protocol](DESESCENE_CHILD_LOCAL_STATE_PROTOCOL.md), [real GPU checks](DESESCENE_CHILD_LOCAL_STATE_GPU_VALIDATION.json) and [CPU validation](DESESCENE_CHILD_LOCAL_STATE_VALIDATION.json).
- [Completed B0/M1, loss500,900-frame paired metrics,64-frame mechanisms and trained Encoder timings](desescene_completed_20261024/README.md). M1 has no demonstrated gain. Encoder wall means are R0 180.05ms, E1 177.72ms, E2 162.00ms and E3 167.26ms on the same GPU3 PyTorch explicitFP16 protocol; these are not full-generation or TensorRT timings.
- [Completed C1 Static FM/BCE evidence and ranking diagnostics](DESESCENE_STATE_CONDITION_PHASE0_RESULTS.md). C1 uses a different scratch/NFE4 cohort. Its metrics must not be mixed with the NFE2 matched-adaptation pilots.
- [Registered next experiments](DESESCENE_NEXT_EXPERIMENTS_20261024.md) and [machine-readable backlog](../../configs/research_v2/hier_occ_flow/desescene_research_backlog.yaml).

Already automatic: finish the two20K pilots,900-frame primary native/continuous evaluation, then64-frame NFE/condition diagnostics. The GPU2 transfer keeps the original GPU3 training process and pinned source identity intact.

Registered but not auto-queued: Encoder-only fine-tuning with frozen Flow, matched Encoder+Flow fine-tuning, conditional scratch retraining. These still require an explicit adaptation runner, strict source identity, frozen-parameter/gradient/resume checks and GPU smoke validation. A plan entry is not a claim of running training. Each new arm remains20K; no automatic100K or additional training seed.

Weights, optimizer checkpoints, full point clouds and compiled caches stay server-local. The linked snapshot includes compact metrics, immutable saved-checkpoint SHA identities, transfer audit and the actual continuation script for reproducibility.
