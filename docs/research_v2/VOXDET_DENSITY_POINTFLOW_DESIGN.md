# VoxDet density + straightened PointFlow

This is an independent route under `fpsgen/voxel_point/`. Existing BEVFlow,
Teacher/Student checkpoints, datasets and BEV evaluation code are unchanged.
The first implementation provides correctness gates and bounded smoke runs;
it does not launch formal training or claim final generation quality.

## Geometry and data

`configs/research_v2/voxel_point/geometry_50m_256x256x32.yaml` is the shared
geometry source: `[-50,50] × [-50,50] × [-4,4.4]`, XYZ grid `256×256×32`,
cell size `0.390625×0.390625×0.2625 m`. This is **2,097,152** cells, correcting
the arithmetic total in the initial proposal while preserving its grid.
Official VoxDet coarse ratio2 yields `128×128×16` cells of
`0.78125×0.78125×0.525 m`. Coarse positions are explicitly XYZ ordered.

The actual point ROI is the open disk `x²+y²<2500` and open height interval
`-4<z<4.4`, about the current LiDAR origin. Voxel validity uses the closest
point of the XY cell to the origin, retaining every cell with nonzero disk
intersection. Tangency alone is invalid. A center outside the disk does not
invalidate an intersecting cell. GT points are rasterized using their actual
XYZ, never a center mask. Sampling conditions a Gaussian on cell ∩ disk;
inverse-CDF truncation and rejection avoid boundary clamping/duplicates.

Dense tensors use XYZ order. Flat IDs are `(iz*ny+iy)*nx+ix`; conversion
transposes XYZ↔ZYX explicitly. Density/counts/centers and coarse features all
use the same spec. `voxel_position_grid_coarse()` exposes the original
preprocessing positional grid for audits; the official LiDAR trunk does not
consume this camera-related positional tensor.

GT is only `gt_possion/<frame>.ply`, exactly180000 finite in-ROI XYZ. Fourth
columns/PLY label properties are ignored. Invalid targets raise a scene-specific
error; no clipping, point cap, replacement, resampling or repetition occurs.
Numeric sequence/frame pairing uses explicit `input_` sibling lookup. Height
bounds are verified against the existing Poisson generator and training
contract, not inferred from LiDiff. See the source audit.

Train:00–07,09,10. Development: numeric frames500–599 per eligible sequence,
with frames480–499 and600–619 excluded from training. Current split:
17870 train,900 development,360 guards; seq04 has no frame500 block.
The complete split manifest lives under the ignored experiment audit directory.
Seq08 was not loaded in this implementation. A test evaluation refuses to
load it until the development threshold/config is marked frozen.

## Stage1

The model instantiates extracted official LiDAR classes with their configured
parameters: the6→64→128→256→128 Point MLP and scatter-max tokenizer;
`CustomResNet3D [2,2,2]` at128 channels; `GeneralizedLSSFPN`; the additional
local `CustomResNet3D [3,3,3]`; the spatially decoupled neck including adaptive
pooling and2D deformable convolutions; six-direction offset prediction; four
`VoxelAttentionAggregation` refinements. Parameterless registry wrappers are
flattened, not replaced by another network. Local batch is1, as in upstream.

Input feature adaptation: relative coarse-cell XYZ, absolute XY,
`norm(XYZ)/50`. The input_ semantic column is never used as intensity. No
camera/depth/calibration/semantic layout is loaded. Main and auxiliary heads
are1-channel, with negative trainable final biases. Eval exposes only
`pred_density [1,1,256,256,32]`. FP32 softplus makes counts nonnegative and
masks invalid cells. There is one network evaluation, no diffusion/FM sampler.

Loss weights are YAML controlled: positive log-count SmoothL1=1,
background=.1, summed2×2×2 coarse consistency=.5, normalized distribution
Mass-TV=.1, relative mass error=.1, occupied-run direction loss=.1,
auxiliary density=.3. Positive/background reductions normalize separately.
All density reductions/log1p/softplus are FP32. Auxiliary supervision has the
same density objective. Semantic CE/scaling losses are not imported into the
active extracted heads. Direction targets are contiguous occupied runs,
normalized by fine grid axis length, masked to occupied ROI cells and
interpolated to the original offset output resolution. This is a new
**density-derived geometric offset objective**, not original semantic VoxNT.

One documented coordinate compatibility patch corrects PyTorch grid_sample's
W,H,D coordinate order for XYZ tensors to Z,Y,X and uses cell-center
normalization for align_corners=False. Width/depth/refinement counts are retained.
`VoxDet-Density-NoOffset` is a later registered structural ablation; it has not
been implemented or trained in this first Full-model delivery.

## Density → source

Finite/nonnegative masked density is normalized in FP64 allocation arithmetic.
Zero/near-zero valid mass raises `InvalidDensityError`. PWM's uniform fallback
is intentionally removed. Integer floor + seeded stochastic residual allocation
with replacement gives exactly180000 points. Counts outside valid cells are0.
The Gaussian std is one quarter of each physical axis size. Inverse-CDF cell
truncation and disk rejection preserve the conditional law even in corner
slivers. Source rows are sorted by voxel ID and stay source-indexed thereafter.

The only bridge is density→integer counts→anchor XYZ + voxel IDs/centers.
No neck/offset/attention maps, hidden volume or learned cross-stage adapters
are exposed. Inference seeds both explicit source sampling and upstream point
serialization/tokenizer RNGs. Manifests record seeds, geometry/config/checkpoint
hashes, source ordering, generated support and synchronized stage timings.

## Teacher, OT and Student

PWM's unchanged official PTv3 trunk/reference adapter/time conditioning are
vendored with namespace-only changes. Metric-scene token hyperparameters are
quantization0.1m, fine0.1m, coarse0.2m, with128/256 channels and3/4/2
fine/coarse/decoder depths. Token grouping is the official architecture
operation; all180k source rows are returned. These token sizes are new
experimental choices, not a claim of unchanged historical PWM defaults.
Teacher receives180k sources plus180k
GT context, known flags0/1 and time0. Only the first180k displacements become
endpoints. Symmetric square-root Chamfer uses the same0.5 factor as PWM/FPSGen.
KeOps KNN selects nearest indices, and selected Euclidean coordinate distances
provide differentiable CD without a dense pairwise matrix. This replaces
PWM's large CUDA-extension requirement and unsafe dense CPU fallback.

Teacher warmup is1 GT-source epoch, then fixed50% mixed sources for a10-epoch
initial budget. Student budget is30 epochs, following100–500 successful-update
pilot and target-quality gates. At fraction.5 the final training driver uses
shuffled GT/predicted source pairs, separately seeded from anchor/time sampling.
Every sample logs its source kind. Predicted caches must carry integer counts,
matching frame/grid, Stage1 checkpoint hash and predicted-density provenance.
Smoke caches are rejected for formal training.

OT reuses existing FPSGen reciprocal KNN/coalescing and sparse log-domain
Sinkhorn kernels. All costs, plans, marginals and barycenters use FP32.
No dense180k×180k matrix is constructed. Refinement starts at Teacher endpoint:
`teacher + alpha*(barycenter-teacher)`. Defaults: K16,epsilon.01,100 iterations,
alpha1; registered training-only grid K16/32,epsilon.01/.02/.05,
iterations100/200,alpha.5/1.

`strict_balanced_ot` requires a finite plan, actual nonzero row/column coverage
and max marginal relative error≤.01 on both axes, then geometry checks.
`sinkhorn_guided_geometry` may accept an unbalanced finite endpoint if CD
improves, coverage@.1 does not fall by>.01, mean spacing retains≥.8 of Teacher,
spacing CV≤1.25×Teacher, ROI violation does not increase by>.001, and max
refinement displacement≤10m. These are initial registered safety thresholds,
not tuned optimal settings. Every rejected candidate falls back to Teacher and
keeps rejection diagnostics. **Current formal mode is geometry-guided**, pending
100-scene mixed-source quality confirmation. Reports never label it converged OT.

Student sees only current XYZ, original voxel IDs/centers and time. For
`Pt=(1-t)P0+tP1†`, target velocity is `P1†-P0`; loss is pure FP32 velocity MSE.
No rollout Chamfer, semantic loss, feature alignment or GT Student input exists.
Euler4/8/16 are registered,8 is default. Inference constructs neither Teacher
nor Sinkhorn. GT is read only by a separate evaluator.

## Precision and memory

Stage1: FP32 checked; FP16 AMP checked; density math FP32. Stage2 training:
FP16 AMP, FlashAttention2.5.9.post1, tensor-only reentrant checkpoints around
complete time-conditioned blocks. Replay preserves both dense and sparse
feature inputs; unused hierarchy parent references are excluded to prevent
C++ autograd/Point cycles. A GPU comparison against unmodified PWM showed
zero forward/gradient error on the dedicated parity test.

SpConv2.3.6 eval bypasses its training autocast wrapper; sparse Teacher/Student
eval is therefore FP32, with official attention internally casting QKV toFP16.
No model width/depth or scene count was reduced. Batch1/gradient accumulation
are supported. KeOps caches are isolated by visible GPU set and rank, avoiding
single-GPU cached device properties under DDP.

## Commands and formal gate boundary

All scripts use --config/--output; --max-frames limits distinct scenes;
training --max-steps counts successful optimizer updates. --smoke is at most
32 updates and its checkpoints/results are marked mechanics-only. Formal
training requires --gate-report with the relevant READY entry. Formal Student
also requires accepted pseudo_target_gate evidence. Scheduler advances only
on successful updates. Successful and attempted frame exposures are separate.

Examples (run from repository root):

```bash
OMP_NUM_THREADS=4 env/voxdet_density_py39/bin/python scripts/research_v2/voxel_point/audit_dataset.py \
  --config configs/research_v2/voxel_point/voxdet_density_stage1.yaml \
  --output experiments/voxel_point/dataset_audit_new --max-frames 100
CUDA_VISIBLE_DEVICES=2 bash scripts/research_v2/voxel_point/launch.sh train_voxdet_density \
  --config configs/research_v2/voxel_point/voxdet_density_stage1.yaml \
  --output experiments/voxel_point/stage1_new --max-steps 16 --max-frames 1 --smoke
```

Fill checkpoint/cache fields in copied YAMLs after each validated stage. Run
`eval_voxdet_density.py --split train --cache-counts` for predicted caches;
`train_mapping_teacher.py`; then `diagnose_sinkhorn.py` on fixed20→100 training
scenes for BOTH source types before formal Student. --grid-search explicitly
runs only the registered24 settings. `sample_pipeline.py` requires an input-only
JSON manifest (`id,input`) and only reads input_/*.npy. Output evaluation uses a
separate (`id,generated,gt`) manifest with `evaluate_final_points.py`.

A=source_anchors.npy; B=GT-density PointFlow oracle diagnostic only;
C=Student trained with sinkhorn:null; D=geometry-guided target Student. B is implemented only in the separate training/development
`diagnose_pointflow_sources.py` entry with an explicit oracle_not_deployable
label; it cannot be added as a deployment mode. C uses
`pointflow_teacher_only_180k.yaml`. C/D require separately trained checkpoints.
`compare_with_bevflow.py` compares identical frame-ID point-evaluation reports.
Use the same evaluator for both exports (CD meters, F-score/coverage fractions).
FD_U remains the existing frozen Utonia protocol: PCA256 and sampling stability
for small manifests, not an estimate of a precise population truth. See
`UTONIA_FD_EVALUATION.md` and existing extract_features/compute_fd scripts.
No seq08 comparison, FD_U claim, Euler ablation or formal A–D quality study has
been run with these short smoke checkpoints.

Stage1 initial formal budget:100000 successful frame exposures =50000 updates
at2 GPUs×B1 without accumulation; AdamW3e-4/weight_decay.01, OneCycle pct_start.05,
checkpoints25/50/75/100%. All thresholds/loss/grid decisions must be frozen on
training/development, never seq08. The current readiness report allows Stage1
formal training but blocks downstream quality-dependent stages. Nothing is
queued automatically.
