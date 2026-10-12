# First implementation results — 2026-10-13

All model checkpoints in this report are **mechanics smoke checkpoints**, not
formally trained research results. Seq08 was not read. No long training queue
was started. Existing FPSGen experiments/checkpoints/evaluation code remain
unchanged. Source hashes, exact as-run checkpoint configs and formal/runtime
YAML hashes accompany this report in voxel_point_reports/.

## Architecture and data

VoxDet commit e10d8280de96d4f03f54b616e57e7c6754044fa4; PWM commit
c6e809486db3d0b6f8503318dfea246c06c7a50e. Official LiDAR module configuration:
22,084,900 parameters; density adaptation22,082,470. All trunk parameter counts
identical. Official PTv3 Teacher14,708,739; Student14,692,739.

Geometry PASS:256×256×32 =2,097,152 cells, range[-50,50]×[-50,50]×[-4,4.4],
voxel0.390625×0.390625×0.2625m, coarse128×128×16, true circle-cell intersection.
100 spatially spaced training GT frames:100 accepted,0 anomalous;180000 XYZ
per frame. Nonempty voxels31,218–126,048, occupancy over valid cells1.878–7.584%,
max occupied count6–20. Per-scene p50/p90/p99 and full count histograms are in
dataset_audit_100.json. These are sample statistics, not full-dataset extrema;
no count cap is used. Train/dev/guards17870/900/360; fixed temporal guard20.

## GPU correctness

All GPU execution was on physical2/3. Peaks below are allocated GiB. Mean step
includes cold startup and checkpoint overhead; steady step excludes first step.

| Run | Successful updates | Peak allocated GiB | Mean s/update | Steady s/update |
| --- | ---: | ---: | ---: | ---: |
| stage1_gpu2_fp32 | 1 | 4.802 | 4.302 | — |
| stage1_gpu2_fp16 | 16 | 4.741 | 1.083 | 0.767 |
| stage1_gpu3_fp32 | 1 | 4.802 | 5.289 | — |
| stage1_gpu3_fp16 | 1 | 4.563 | 3.916 | — |
| stage1_ddp_fp16 | 32 | 4.823 | 0.941 | 0.773 |
| teacher_gpu2_final | 16 | 9.005 | 2.739 | 1.419 |
| teacher_gpu3_mixed | 16 | 9.852 | 2.812 | 1.441 |
| student_gpu2_mixed | 16 | 5.401 | 3.048 | 2.187 |
| student_gpu3_mixed | 16 | 4.975 | 3.003 | 2.179 |
| student_ddp_isolated_cache | 32 | 5.464 | 3.018 | 2.193 |

Every final smoke gradient is finite. Stage1 GPU2 FP16 loss1.20037→0.55899;
Stage1 DDP rank0 loss1.20037→0.47903. Teacher GPU2 CD loss0.22045→0.11959.
Student GPU2 MSE0.08165→0.01345. Final Student DDP rank0
MSE0.07811→0.01183, rank1 0.08167→0.01514. These losses verify regression
mechanics, not held-out geometry quality.

Teacher mixed smoke saw10 GT/6 predicted source draws over16 updates. Initial
single-card Student stress smoke saw2 GT/14 predicted draws (randomized50%
probability). The final driver and final DDP checkpoint use independently
shuffled source pairs: exactly16 GT/16 predicted per rank over32 updates.
Each rank accepted16 GT-source refined targets and fell back on16 predicted
sources. This fallback is explicit, not an assertion that D is quality-ready.

Numerical checkpoint parity versus unchanged PWM: max forward difference0,
max parameter-gradient difference0, finite. This512-point parity check is
additional to actual180k+180k Teacher and180k Student gates, not a replacement.
Compatibility failures and their corrected evidence are retained separately.

## Actual sparse OT

The20-scene K16 GT-source diagnostic uses seq00 frames0–19 from the training
manifest. It is a fixed mechanics subset, not a representative100-scene formal
selection set. Teacher is the short mixed-source smoke checkpoint.

| Endpoint | Mean sqrt-CD m | Mean coverage@10cm | Mean spacing CV |
| --- | ---: | ---: | ---: |
| source | 0.134607 | 0.248541 | 0.502443 |
| teacher | 0.126501 | 0.284931 | 0.537806 |
| nearest_gt | 0.042226 | 0.585437 | 1.553137 |
| knn_barycenter | 0.128905 | 0.279291 | 0.678925 |
| sparse Sinkhorn candidate | 0.082871 | 0.653187 | 0.451709 |

GT-source geometry-guided acceptance20/20; balanced convergence0/20.
Max-row relative marginal error range
0.190–0.428,
far above.01. K16 runtime mean0.825s,
peak allocated0.574GiB.
Nearest-GT has lower CD but worse spacing CV and lower coverage than Sinkhorn;
CD alone does not justify projection. These observations still require proper
trained Teacher and the formal20→100 diagnostic before method selection.

Predicted-source K32 (two real180k scenes):

| Frame | Teacher CD m | Candidate CD m | Row relative error | Max displacement m | Geometry accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| 000000 | 0.317325 | 0.118285 | 799.062 | 26.586 | NO |
| 000001 | 0.317391 | 0.118306 | 565.319 | 25.614 | NO |

Both plans finite; actual graph and nonzero plan row/column coverage PASS.
Balanced convergence FAIL; column relative errors≈6.4e-5 while row errors are
565–799. Runtime1.18–1.43s, peak≈1.07GiB. Candidate CD improves, but max
refinement displacement25–27m exceeds the registered10m gate, so both return
Teacher endpoint. Predicted-density source gap remains a **formal blocker**.
No dense NxN cost was allocated. Strict/geometry policy labeling and invalid
plan fallback are covered by CPU tests; actual marginal failure is not hidden.
The100-scene formal quality gate is pending, not reported as PASS or silently
replaced by this20-scene smoke.

## End-to-end and separate evaluation

Final20 real input_ frames (including the five-frame Gate E subset) PASS.
Each uses exactly one VoxDet network evaluation, seeded density allocation,
180000 valid-cell/disk Gaussian anchors and Euler8 PointFlow. Every final
file is finite float32[180000,3]. Deployment input manifest has onlyid/input;
Teacher, Sinkhorn, GT counts/context/semantics and Stage1 hidden features are
absent from the inference call graph. GT is read only by a separate evaluator.

Warmed timing over frames1–19, physicalGPU3, same compatibility environment:
- stage1: 144.295ms/frame.
- allocation_and_anchors: 22.970ms/frame.
- pointflow: 2553.372ms/frame.
- total: 2720.637ms/frame.

Peak allocated inference2.984GiB. First cold frame
was6250.5ms and is excluded above.
Timing includes synchronization and source sampling, excludes checkpoint load,
PLY/NPY IO and result serialization. No comparison with BEVFlow latency is
claimed without a matched export/run. PointFlow currently dominates latency.

Separate20-frame smoke evaluation (same final checkpoint, no quality claim):
CD0.366903m;
F-score@10cm0.153104,
@20cm0.542242;
coverage@10cm0.140396,
@20cm0.496275;
out-of-ROI ratio0.000289.
F-score/coverage are fractions, CD uses half of the two Euclidean means.
This short-run output is not a high-quality trained result. Final points are
not silently clamped/repeated to improve metrics. The trained model must later
satisfy quality checks. seq08/FD_U/Euler4/16/C-vs-D formal quality remain pending.

## Ready for formal training

Stage1 READY for its requested supervised budget. Teacher BLOCKED pending
trained Stage1 and validated predicted-source cache. Sinkhorn BLOCKED because
predicted-source candidates fail acceptance, balanced OT does not converge,
and trained20→100-scene gate is incomplete. PointFlow formal training BLOCKED
pending reliable mixed-source pseudo-targets and100–500-update pilot.

CPU tests18 PASS, both environments pip check PASS, scripts compile and shell
syntax checks PASS. No checkpoint/dataset/cache/env/compiled artifact is in the
commit. The executable source, YAMLs, selected source provenance and small
reports are committed separately from all pre-existing local modifications.
A separate GT-density Oracle B entry was smoke-tested on development00/000500;
it is explicitly labeled oracle_not_deployable and never enters sample_pipeline.
Its short-model results are recorded in oracle_development_smoke.json.
No long training or evaluation queue was started; all bounded smoke processes
finish before handoff.
