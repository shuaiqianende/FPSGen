# Formal Stage-1 BEV method comparison

## 1. Selection rationale

The fixed main-table candidates are FPSGen Legacy BEVFlow, DiC-S-BEV (2025),
SiD2-style Residual U-ViT-S-BEV (2025), and DiP-S/16-BEV (2026). These are
architecture adaptations to pixel-space BEV, not selections based on which
baseline finishes weakest. No new backbone is introduced in this study.

PixelU is excluded because its original clean-data prediction training recipe
is not the same as the existing direct-velocity adaptation. PixelDiT is an
existing custom Small scaling with unresolved short-run optimization and
condition-use issues, not a main-table candidate. Historical NCSNpp results
remain available as a convolutional anchor; NCSNpp is not queued here.

## 2. Upstream method correspondence and source audit

- DiC: [official source at bfa541b5](https://github.com/YuchuanTian/DiC/blob/bfa541b5e3a968919f3a399fc0e223e877439b3b/dic_models.py).
  Hidden96, depth6/6/5/6/6, multipliers1/2/4/2/1, sparse skip stride3,
  overlapping3×3 embedding, PixelUnshuffle/PixelShuffle, GroupNorm/GELU
  residual blocks and modulation are retained. Class embeddings and the
  variance output are intentionally omitted: FPSGen predicts3 velocity
  channels with spatial BEV conditions. Formal configs explicitly request
  `dic.final_norm: layernorm2d`, matching the official non-affine channel
  LayerNorm2d with eps1e-6, including flattened-Linear patch initialization.
  Old configs/checkpoints lacking this key retain historical GroupNorm and
  their original initialization path; they are not relabelled corrected DiC.
- DiP: [official source at 949294f2](https://github.com/NJU-PCALab/DiP/blob/949294f290f2380a7767649f343679e24dab4410/src/models/transformer/dip.py).
  Patch16, width384,6 heads/head-dim64,8 FlattenDiT blocks, RMSNorm,
  QK normalization,2-D RoPE and SwiGLU are retained, together with the
 16→8→4→2→1→2→4→8→16 Local Detailer. Use the existing generic BEV
  patch/global adapter; no native-condition ablation is mixed into this run.
- SiD2: **paper-based scaled implementation**, not official source release.
  Patch2, channels64/128/256/384, E3-D3, middle16; level-wise residual
  `high + up(low - down(high))` skips, not block-wise concatenative U-Net.
  The existing separate-first pyramid and per-resolution modulation are used.
  Its local transformer norm/FFN are the existing custom implementation, not
  a claim of full official natural-image architecture numerical equivalence.

## 3. Shared training contract

All methods predict velocity from3×256×256 `[D,H,M]` BEV under linear FM:
`B0 ~ N(0,I)`, uniform `t`, `Bt=(1-t)B0+tB1`, `u=B1-B0`;
loss is the mean squared velocity error weighted `[1,2,1]`.
Train sequences00–07/09/10, held-out08, `gt_possion`,180000 GT points,
range50m, identical Y-flip augmentation and32-channel DynamicKNNPillarEncoder
(K16), vehicle1 and road1, eight equally probable condition states.

Train from scratch, seed42, AdamW1e-4, betas(.9,.999), weight decay1e-4,
warmup1000 optimizer steps, cosine using `trainer.estimated_stepping_batches`.
No REPA, teacher, perceptual/cycle/frequency loss, altered representation,
time sampling, frontend, model widths or depths.

## 4. Runtime and gate sequencing

Only physicalGPU2+GPU3 are available for this study. One training model owns
both GPUs; per-rank batch16, global32 (the same smaller final batch as historical
Legacy is retained; observed local size13 is recorded, not relabelled16).
Geometry/frontend/targets/loss remain FP32, dense adapter/backbone uses FP16
autocast and Inductor unless an explicit backend exception applies. The existing
PyTorch2.0.1+cu117 environment is not upgraded. Worker4 per rank, prefetch4,
pinned/persistent workers, deterministic initialization, zero-copy NumPy,
equivalent fast Y flip and asynchronous tensor transfer are identical.

New runs uniformly initialize GradScaler at8192, with its original dynamic
growth/backoff rules and no clipping. The default65536 caused first-step
overflow in DiC/DiP diagnostics;8192 passed their finite-gradient checks.
This is a documented numerical runtime setting, not a change to AdamW, LR,
FM loss or masks. Previously completed Legacy used the historical default
scaler setup; its exact initial/skip telemetry was not saved. New gates and
pilots revalidate this common setting rather than claiming old tests used it.

Production uses no visualization or per-batch CUDA synchronization profiler.
The lightweight CSV monitor reports wall-clock throughput; host-ready gaps
in formal manifests **are not pure disk/DataLoader latency**. A400-step DDP
pipeline gate excludes the first100 startup steps and checks the next300
for recurring >2s host gaps. Such evidence alone cannot attribute a stall to
storage. Each candidate then runs B16 FP16 eager16, B16 Inductor16,
DDP B16×2 for32 steps, followed by a10-epoch correctness pilot.

All pilots finish and pass condition100/111 B100 and B20 CFG1/2 before any
new formal500-epoch run starts. Condition Gshuffle<.005 is an inspection
stop, **not a performance-based elimination or ranking criterion**. OOM,
nonfinite gradients/loss, compile errors, serious stalls or a failed gate
stop the queue. No automatic B8, width change, eager fallback, candidate
replacement or resume occurs. Activation checkpointing/backend exceptions
require a separately recorded adjustment and revalidation if needed.

Every formal run receives a fresh32-step preflight before fresh500-epoch
initialization. Checkpoints are saved every50 completed epochs and at final;
the10-epoch pilot's final checkpoint is explicitly saved even with cadence50.
Rank manifests include parameter partition, config/git dirty status, batch,
memory and skipped optimizer updates. GradScaler skipping is counted from its
actual unscale-time overflow flags, accumulated on-device and materialized
only every100 steps; no per-batch scale `.item()` polling is introduced.

## 5. Parameter partition

| Method | Core | Condition adapter | PointPillar | Total |
|---|---:|---:|---:|---:|
| FPSGen BEVFlow |21,498,179|2,032,736|1,600|23,532,515|
| Corrected DiC-S-BEV |32,157,891|236,741|1,600|32,396,232|
| SiD2-style-BEV |44,114,188|360,968|1,600|44,476,756|
| DiP-S/16-BEV |39,038,467|3,489,794|1,600|42,529,861|

Legacy condition counts include its pyramid and stage fusion projections;
transformer cross-attention remains core. All partitions sum exactly to total.
Counts include non-trainable parameters; DiP has98,304 frozen positional-
embedding parameters. Manifests separately record total trainable parameters.

## 6. Existing Legacy500-epoch reuse

Final `epoch=499.ckpt` has epoch499/global-step299000,598 steps/epoch,
2GPU,16/rank, seed42,500 epochs, the required dataset/GT/frontend,
FP16+Inductor, optimizer and schedule. It is eligible for task-protocol
reuse and is not retrained. Its historical save cadence was every epoch,
later pruned to every10; new models save every50. Checkpoint I/O differences
must be distinguished from steady-state core speed.

Legacy's initial dirty-tree source hash, full scaler skipped-step count and
complete original rank-wise performance metadata were not recorded. They
remain **unknown**, not retrospectively declared zero/PASS. A reuse manifest
records this limitation. Final quality is recomputed under the frozen formal
evaluation protocol. Legacy final200 recorded train-loss mean is0.0814844;
final500 mean0.0814806, final epoch mean0.0812587. These are train losses,
not the formal generation comparison.

## 7. Final B200 quality and condition utilization

`configs/research_v2/formal_bev_compare/seq08_b200.txt` is frozen before model
evaluation using deterministic uniform indices over sorted complete GT/input
pairs across all sequence08. All final checkpoints use exactly these frames,
Euler10, condition100, exact per-frame noise seeds20261001/20261002/20261003,
CFG1 andCFG2. Pilot B20/B100 are separate and never headline paper results.

Quality reports IoU, occupancyF1, completionF1, height GT-occ MAE and density
Mass-TV. Completion is GT occupied and input-LiDAR unobserved. Condition
evaluation preserves FP32 GT/features, freezes x0/Bt/t, and corrupts conditions
using a fixed non-self frame offset17. It reports100/010/001/111 by five
time-bin centers .1/.3/.5/.7/.9 and headline LiDAR<.2/<.4 and all<.2.
This is a fixed-time quadrature diagnostic, not continuous-bin Monte Carlo.

## 8. Efficiency and statistics

Train throughput is global samples/s from the first consecutive300 steady
steps after100 startup steps, not single-rank speed. Memory is the maximum
of two rank peaks. Startup-inclusive host intervals are distinct from pure
compile time; do not label an unisolated number as compile-only duration.
Single-GPU RTX3090 inference benchmarks B1 andB8, Euler10/CFG2/condition100,
20 warmup and100 timed generations, including frontend but not Dataset/I/O.

For final quality, average the three seeds **within each frame**, then use
10000 paired frame bootstrap resamples, seed20261001. Report FPSGen-minus-
baseline mean delta/95%CI/fraction improved. Positive IoU/F1 differences favor
FPSGen; negative height/Mass-TV differences favor FPSGen. No overall weighted
score or test-selected checkpoint is used. The final epoch500 is mandatory.

## 9. Operations and current evidence

Launch a fresh campaign: `bash scripts/run_formal_bev_compare_gpu23.sh
--output-root outputs/research_v2/formal_bev_compare/revalidated_gpu23_20261003`.
Monitor the master: `tmux attach -t fpsgen_formal_bev_compare`.
Evidence is under `outputs/research_v2/formal_bev_compare/`; queue status,
stage console logs, manifests, epoch/loss CSV, pilot/final evaluation are
separate. The script refuses occupied GPUs and existing stage directories.
Do not rerun it over existing evidence or mistake a tmux shell for success.
An explicit fresh retry can use `--output-root
outputs/research_v2/formal_bev_compare/revalidated_gpu23`; the queue repeats
all gates and never resumes a pilot/checkpoint or overwrites failed stages.
Selected DiC actual configs use the correctly named `_eager_checkpointed`
files. Its requested `_inductor` configs are retained as audited requests,
not falsely reported as successful compile configurations.

The CPU verification suite passed55 tests, including corrected DiC final
normalization, activation-checkpoint equivalence, bounded preflight checks,
actual GradScaler overflow accounting, frozen manifests, and failure
propagation from the dedicated tmux worker.

The master freezes Python-source hashes and selected config snapshots at
launch. A source edit stops the queue before the next stage instead of mixing
implementations. Each500-epoch run executes inside its own tmux session:
`fpsgen_formal_dic`, `fpsgen_formal_sid2`, `fpsgen_formal_dip`; only one
training process group uses GPU2+3 at a time. The master also streams each
run's log and waits for its actual worker exit status. Failed stages do not
advance to the next model. No worker environment secrets are serialized.

The initial suite passed33 CPU tests; expanded tests also verify checkpoint
equivalence, unscaled-gradient checking and actual overflow accounting. GPU
pilots/formal results remain pending until their actual stage outputs exist.
Only Legacy500 is completed.
The small [gate evidence snapshot](formal_bev_gate_evidence.json) records
actual rank-wise preflight memory/skips and the initial monitor caveat.
`scripts/summarize_formal_bev_compare.py` produces separate quality,
condition-use, efficiency CSVs and paired uncertainty once results exist;
missing runs are explicitly listed rather than fabricated.

### Initial GPU gate evidence (2026-10-03)

The first Legacy DDP pipeline gate completed400 steps. Timed intervals200–400
cover300 steady steps and9600 samples in122.98155s: **78.06 global samples/s**,
409.94ms/optimizer-step. Rank0 host-ready gap p50/p95/p99/max is
4.32/6.35/11.34/31.85ms; rank1 is2.32/5.84/8.86/13.90ms. No multi-second
host-ready gap was seen. This remains a host-gap observation, not a claim
of pure disk wait. Peak allocated17344.55MiB and reserved18434MiB/rank.

The initial CPU optimizer post-step counter did not observe executed updates
under this integration, so its reported400 skipped steps is invalid, **not**
400 real overflows. Raw manifests were preserved with a separate
`diagnostic_annotations.json`. The corrected monitor reads the scaler's actual
unscale-time found_inf flags. Pipeline gate must be repeated with that monitor
before formal launch.

DiC FP16 eager B16 failed at the first forward: requested192MiB with22.01GiB
already allocated on a23.56GiB RTX3090. Normal Inductor B16 also OOMed at a
decoder allocation. The master queue stopped; no model width or batch was
reduced. Per-block activation checkpointing has bitwise-equal CPU outputs and
gradients and makes eager B16 fit. Checkpointing with Inductor fails in this
PyTorch2.0 environment: first an RNG-state tracing assertion, then, after
omitting unnecessary RNG preservation in deterministic blocks, a FakeTensor
GroupNorm-weight conversion error. No errors were suppressed and no framework
upgrade was attempted.

Eager/checkpoint B16 diagnostics with explicit scaler initial scales1024 and
8192 completed16 finite-gradient steps with actual optimizer updates, but
the existing zero-initialized condition gates had not opened nonzero feature-
encoder gradients within16 steps. The strict preflight therefore remained
FAILED, not a claimed condition-path PASS. Gate parameters themselves moved
away from zero. The user explicitly approved changing the already-supported
gate initial value to0.1 and repeating preflight. Both formal/pilot configs
now record0.1; architecture, condition sources and all task/optimizer settings
are unchanged. Gate0.1 eager/checkpoint single16 and DDP32-step preflights
both passed with finite core/condition gradients and zero scaler skips, using
an explicitly recorded initial GradScaler scale8192. The selected actual
configs are `dic_s_500ep_ddp2_b16_fp16_eager_checkpointed.yaml` and
`dic_s_10ep_pilot_eager_checkpointed.yaml`. Their explicit backend exception
references the preserved failed compile stage; requested Inductor configs are
retained for audit. Quality remains comparable under the shared task contract,
but the efficiency table must not label DiC as an Inductor run.

SiD2 B16 FP16 eager and Inductor single-GPU16-step preflights passed with
finite loss/core/condition gradients and no OOM. Its DDP32-step gate also
passed on both ranks at16/rank, global32. DiP's first eager B16 preflight
reached backward without OOM but overflowed at the default initial scaler;
an explicit scale8192 eager16 diagnostic passed with active condition
gradients. Inductor single16 and DDP32 also passed with zero skips; its
DDP peak allocated22,235.12MiB is close to the3090 limit, so the full-data
pilot remains mandatory. These gates are separate diagnostics,
not authorization to skip DiC's failure
and start formal training. Full10-epoch pilots and new500-epoch runs have
**had not started at the preflight snapshot**.

### Budget stop: 2026-10-03

The fresh `revalidated_gpu23_20261003` campaign passed the400-step Legacy
pipeline gate, then repeated the candidate preflights and started the DiC
10-epoch pilot. The user rejected its compute cost. The pilot and master
queue were interrupted at07:28 UTC; GPU2/3 were verified released, and other
GPU jobs were untouched. No new500-epoch formal run was started. Do not
automatically restart this queue or move to another long training run.

The latest two complete100-step DiC pilot intervals took201.89s and202.02s,
or about2.02s/step and15.85/15.84 global samples/s. This was FP16 eager with
all-block non-reentrant activation checkpointing, not Inductor. The Legacy
gate's three post-warmup100-step intervals achieved75.98/76.18/76.37 global
samples/s. These are different architectures/runtime configurations, not
evidence that the Legacy training pipeline itself regressed fivefold.

Both DiC GPUs were sampled at100% utilization and about348W, but utilization
alone does not identify the slow operator. The compile failure specifically
reported a non-FakeTensor parameter dtype conversion in the custom GroupNorm
path. Neither the fraction attributable to recomputation nor the attainable
compiled DiC speed has been isolated. Treat both as runtime issues requiring
a tightly bounded operator-level diagnostic, not proof that the upstream
DiC architecture inherently requires seven days. Any further long training
needs an acceptable measured throughput and revised compute-budget approval.

### Bounded SiD2 / DiP speed check: 2026-10-03

At the user's request DiC was deferred. Only SiD2 and DiP were tested,
serially on physicalGPU2+3,16/rank (global32), FP16 dense Inductor, eager
FP32 geometry and unchanged `gt_possion` data/180000-point task. Each fresh
run had a hard200-update limit and no pilot/formal continuation. Expensive
finite-gradient checks were restricted to the first16 updates. Both ranks
passed finite core/condition gradients with nonzero condition-encoder
gradients. Both models performed200 actual updates, with zero scaler skips
and no OOM. GPU2/3 were verified released after the two stages finished.

| Model | Stable ms/update | Global samples/s | Peak allocated GiB | Host gap p99 / max ms | Approx min/epoch |
|---|---:|---:|---:|---:|---:|
| SiD2-style Residual U-ViT-S-BEV |317.10|100.91|9.16|13.53 /58.82|3.16|
| DiP-S/16-BEV |250.10|127.95|21.71|8.79 /29.65|2.49|

The100-update timed interval excludes startup; the retained historical
lightweight logger's `global_step` label has a+1 offset, so CSV boundaries
100..200 correspond to actual updates100..199 (99 startup updates excluded,
last update200 also outside timing). The summary records this explicitly;
no extra GPU run was needed to correct a label. Host gap is still a framework
batch-end/start gap, not pure storage latency. Peak memory is the maximum
across both ranks and includes startup. SiD2 captured4 graphs with2 known
`DICT_UPDATE` diagnostic-dictionary breaks; DiP captured1 graph without
breaks. Neither observation is a generation-quality ranking.

The10-epoch compute-only projections are about32min (SiD2) and25min (DiP);
500-epoch projections are26.3h and20.8h, not an authorization or launch.
These estimates assume the sampled speed persists and exclude startup,
checkpoint/evaluation cost, last-batch recompilation and long-term I/O
variation.200 steps did not exercise an epoch boundary or every point-count
tail; DiP's small startup memory margin still requires caution.

Evidence: `outputs/research_v2/formal_bev_compare/sid2_dip_speed_20261003/`
contains the separate200-step manifests, preflight checks, CSV timings,
`throughput_summary.csv` and `throughput_summary.json`. No10-epoch pilot or
new500-epoch run was started by this diagnostic. The tmux session
`fpsgen_sid2_dip_speed_gpu23` retains console results for inspection.

### Authorized SiD2 / DiP training campaign

After the speed checks the user authorized training, with DiC still deferred.
The explicit `--defer-dic --methods sid2_s dip_s16` scope preserves the
original fail-stop policy: it is not an automatic fallback after a failed
candidate. The reused400-step Legacy data-pipeline gate and eager16/
Inductor16/DDP32 gates for both selected models are accepted only after
checking every `fpsgen/` Python source hash, model/data/loader/runtime
controls, actual updates, zero skips and both-rank gradient PASS records.
The orchestration script may change without rerunning unchanged model
compute. This reuses evidence, not weights or optimizer state.

Execution order: fresh SiD2 pilot10 + B100/B20 checks, fresh DiP pilot10 +
B100/B20 checks, then fresh SiD2 formal500 and final evaluation, fresh DiP
formal500 and final evaluation, followed by final Legacy evaluation.
All pilot gates must pass before the first new500-epoch run; formal runs
never resume pilot weights. New runs use GPU2+3 serially,16/rank, seed42,
FP16 dense Inductor, `gt_possion`, AdamW1e-4/warmup1000/cosine, and cadence50
with a final checkpoint. Repeated multi-second host gaps, excessive scaler
skips, OOM, NaN or process errors stop the queue without changing configuration.

Master: `tmux attach -t fpsgen_formal_sid2_dip_gpu23`.
Formal worker sessions (created only after pilot gates):
`fpsgen_formal_sid2` and `fpsgen_formal_dip`.
Evidence root:
`outputs/research_v2/formal_bev_compare/sid2_dip_formal_20261003/`.
The frozen plan records DiC as deferred. Final summaries retain the missing
DiC entry and distinguish completion of this selected scope from completion
of the full original four-method main table.

## 10. Limitations

DiC, DiP and SiD2 are adapted to the common FPSGen BEV Flow-Matching objective
and condition interface. Results compare their architectures as Stage-1 BEV
velocity backbones rather than reproducing each method's original natural-
image training recipe. The SiD2-style model is a paper-based Residual U-ViT
implementation rather than an official released SiD2 codebase.

This main study has one train seed42, with3 generation seeds/frame; those
generation seeds do not replace repeated training seeds. Report generation
quality, condition use, training efficiency and inference efficiency separately.

The historical FlowIMG noise/time/mask streams are separately seeded but
start with the same seeds on both ranks. Thus ranks process different frames
while their sampled noise/time/masks are correlated. This policy is retained
for comparison with the completed Legacy, not silently changed or claimed to
be32 independent stochastic probes per step. A future rank-independent RNG
study would require a newly matched Legacy run, not reuse of this checkpoint.
