# MFV2-A latest-checkpoint interim41 evaluation

User requested a mid-training inference check. This adds a separate evaluator without modifying either running training worktree or weakening the original final20K evaluation gate.

- Immutable checkpoint snapshot:7500 successful updates /15000 successful frame exposures /2AMP skips; EMA weights. SHA256 `a3fd96045876c9f39f3fb6660d44630b6483c9d231080557b51a76fb5b81d8bc`.
- Snapshot is a hardlink to the inode atomically published by the trainer (`os.replace`), so later `resume_latest.pt` replacements do not change the sampled model. Checkpoint is local/ignored, never committed.
- Original A source SHA `37ddfcc699e22e6ba9b60237968c65304ccaf9eb64fc4e5dc94b6eebde164378`; checked by the original source-hash function in the original worktree. Model/math/vendor files and original final evaluator are byte-identical.
- New evaluator checks exact intended intermediate counters, scheduler, configuration, source identity and geometry; requires 0<updates<10000 and never labels this as final20K.
- seq08 skip10041 frames0..4000; EMA; NFE1/2/4/8; fixed0.5; original noise seeds, frozen rawGT/input hashes; all GT included. Coarse Parent voxel centers, no Stage2 and noJitter.
- Full P/R/F1/IoU/count/Ceiling, Fixed-K84963/120000, exact CUDA KD-tree CD, corrected physical IoU0.5/0.2/0.1, JSD; trajectory discrepancy and exploratory threshold sweep retained. No best threshold is selected.
- GPU0/1 were occupied by another user's training. Evaluation runs on GPU2 sharing **only with this project's A**; verifies the existing PID/owner/command and >=8GiB free. PyTorch allocator cap6GiB,2 CPU threads. Original A and B training continue. Shared runs make no latency claims.
-54 CPU math/protocol/regression tests passed, including8 new partial-checkpoint/resource tests. Final20K checks remain unchanged.
- The finalizer requires164 unique(frame,NFE) rows and all41 registered frames before writing the comparison report. Ordinary FM and originalMF reference checkpoints have20000 exposures; comparison to the interim15000-exposure model is descriptive, not matched-budget evidence or a final GO.

Commands (fresh output only):

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -u \
 scripts/research_v2/meanflow_v2/evaluate_interim.py \
 --config /data-12/M2024-HWZ/FPSGen/worktrees/desescene-meanflow-v2/configs/research_v2/hier_occ_flow/meanflow_v2_a_gpu2.yaml \
 --checkpoint <immutable checkpoint_snapshot.pt> --expected-updates 7500 \
 --training-pid <verified live A pid> \
 --training-root /data-12/M2024-HWZ/FPSGen/worktrees/desescene-meanflow-v2 \
 --output <new quality41 output>
```

Use `--expected-updates 7500` (separate argument). Raw logs/weights remain under `experiments/desescene_meanflow_v2_interim/a_update7500`. The compact final report/CSV are generated only after complete evaluation. An early sandbox-local CPU finalizer launcher produced no result; a host-visible watcher was registered instead, with both launch records preserved. The GPU evaluator was not restarted.
