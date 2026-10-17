# Stage2 FM session exit and bounded learning diagnostic

The original `fpsgen_hs_stage2_384_fm_gb4_gpu23` session exited because its
400-update overfit budget completed and its quality checker returned exit2.
This was **FAIL_ORACLE_OVERFIT**, not an OOM, lost checkpoint or completed
full-dataset pilot. The old shell then exited, removing the tmux session.

The original evidence, logs and checkpoints remain unchanged. At update400,
the first20/last20 mean FM losses were1.98780/1.47091; AMP skipped0 attempts.
Twelve training scenes and two fixed sampling seeds produced:

| Velocity NFE | Macro F1 | Macro IoU |
|---|---:|---:|
|1|0.563938|0.392999|
|2|0.570165|0.399119|
|4|0.572637|0.401563|
|8|0.573666|0.402582|

These are **Oracle training overfit diagnostics**, not predicted-parent or
test-set performance. All-candidate-children-occupied F1 was0.563801. The
registered gate requires an absolute0.02 F1 margin; the observed0.009865
margin failed. The loss criterion passed. No threshold was relaxed.

Update200 NFE8 F1/IoU was0.562984/0.392069. With continued improvement, finite
gradients, synchronized weights and zero AMP skips, a bounded additional
600-update learning diagnostic is justified before a full-training-set pilot.
This does not establish that the model can overfit; it tests that possibility.

## Explicit continuation protocol

- Same12 Oracle training frames; no seq08 access; same geometry and architecture.
- Same pureFM objective, Gaussian source, fixed evaluation seeds and thresholds.
- GPU2/3 DDP, two complete variable-length scenes/rank, global batch4.
- Strictly continue own update400 online/EMA/AdamW/scaler and both rank RNGs.
- Source checkpoint SHA256:
  `9543a671e3f2e8781d3a71a83707a2d8ee1bb251bc733b3630b53c8ed8910d5c`.
- Extend only the cosine horizon from400 to1000 absolute updates, preserving
  baseLR2e-4 and warmup20. Since the old cosine reachedzero, this explicitly
  raises boundaryLR to about1.34537e-4; it is not an identical-schedule resume.
- New config: `stage2_384_fm_gb4_overfit1000_diagnostic.yaml`.
- New output: `experiments/hier_occ_flow/stage2_384_fm_gb4_overfit1000_diagnostic`.
- Log: `experiments/hier_occ_flow/stage2_384_preflight/fm_gb4_overfit1000_diagnostic.log`.
- Re-evaluate the unchanged quality gate at completion. No full-split pilot,
  BCE job or100K formal training is queued by this diagnostic launcher.

The new tmux uses the same requested name, a persistent shell after completion
and `remain-on-exit`. An exited job remains visible with its real exit status;
the existence of the session is never used as proof that training is running.

CPU regression before launch:34 passed (`test_stage2_384.py` and
`test_stage2_batching.py`); launcher passes `bash -n`. GPU admission recorded
both GPUs idle with no compute processes before starting. Current actual launch
and checkpoint evidence is saved beside the diagnostic logs.

Actual continuation checkpoint verification: update420, AMP skips0, 169 model tensors changed since the400 transition checkpoint, online/EMA rank hashes match,169 AdamW states restored. tmux stays alive, and both ranks run on physical2/3. This is a launch snapshot; see the JSON audit timestamp and live log for progress.
