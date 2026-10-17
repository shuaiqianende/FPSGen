# HS-OccFlow Stage2: 384 parents, 8 children

Implementation branch: `research/hs-occflow-384-stage2`, based on
`5985fb93d6a6e5671877cb3bff8ec371612b290c`.
Actual full-scene GPU2/GPU3 checks passed using final Stage1 weights.
DDP restore passed; learning gates are tracked in the pilot report. Engineering
success alone is not a quality conclusion.

## Fixed representation

| Property | Value |
|---|---|
| Parent grid, XYZ | 384 × 384 × 32 |
| Child split | 2 × 2 × 2 |
| Fine grid, XYZ | 768 × 768 × 64 |
| Bounds | [-50,50] × [-50,50] × [-4,4.4] m |
| Sphere | radius < 50 m, at the LiDAR origin |
| Parent size | 0.2604166667 × 0.2604166667 × 0.2625 m |
| Fine size | 0.1302083333 × 0.1302083333 × 0.13125 m |
| Parent ID | ix + 384 × (iy + 384 × iz) |
| Child bit | dx + 2 × (dy + 2 × dz) |
| Feature/score layout | BCZYX / ZYX |
| Geometry SHA256 | c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b |

GT is raw-map, pre-Poisson **surface-hit occupancy**, data version
`hs-occ-surface-hit-v1`. Sparse parent IDs and uint8 child masks are reused
from the complete audited 384 dataset. No new dense fine labels are created.
Cells intersecting the valid sphere remain eligible; each child has its own
validity mask. Unobserved input cells are not labelled free space.

## Network

The frozen actual Stage1 Occupancy encoder yields three 128-channel volumes.
World-coordinate trilinear sampling produces 384 features per parent. Stage1
occupancy endpoint scores select predicted candidates; they are explicitly
uncalibrated proxies. The encoder runs once per frame, shared across all NFE.

Each parent has one token. Independently projected inputs are the 8D state,
384D sampled LiDAR features, normalized parent XYZ, scalar Stage1 score,
8D input-only observed-child mask, and 32D scene-time embedding. Fusion width
is 64. Parent-centered features are retained; there is no child-aware network.

PTv3 has encoder widths [64,128], depths [2,2], decoder width [64], depth [2],
patch sizes [64,128], absolute parent grid coordinates, deterministic
serialization, and six time-modulated blocks for FM. Pooling/unpooling returns
one feature per original parent. The velocity head outputs [M,8].

Actual CPU construction gives **2,677,128 FM parameters** and **2,558,040 BCE
parameters**. The 119,088 difference belongs to FM state/time modules; the
spatial backbone capacity is matched. Stage2 weights initialize randomly.
No Teacher, Sinkhorn, PointFlow, Shortcut, or auxiliary surface loss is used.

## Objective and candidates

FM uses x0 ~ N(0,I8), x1 = 2O−1, xt = (1−t)x0 + tx1 and masked velocity MSE.
One scalar t is shared by all parents in a scene. BCE is a separate direct
classification baseline, with no time or noise input. Euler NFE 1/2/4/8 use
the same initial noise seed and frozen conditions. Fine threshold defaults to
zero endpoint state; threshold selection belongs to development only.

Oracle support contains all GT-positive parents plus 25% sampled neighboring
empty parents from a 3×3×3 expansion. Neutral scores equal 1 for both classes;
an alternate policy queries actual Stage1 scores for both. Membership is never
an input feature. Predicted support uses only actual Stage1 scores with explicit
threshold/top-K/expansion/budget policies. Overflow raises by default.

Evaluation compares against the full immutable fine GT, including every fine
FN in a Stage1-missed parent. Oracle quality is never deployment quality.
Macro/micro precision, recall, F1, IoU, distance bins, observed/unobserved
regions, child-count MAE, empty-parent suppression, surface fragments and
cross-parent contact edges are implemented.

##384-only numerical fixes verified on GPU

The independent Stage2 wrapper canonicalizes parent order before the input MLPs
and uses the same B1 instance mean/variance formula with scene reductions rather
than atomic index_add. Parameter counts and shapes are unchanged.

SerializedPooling is explicitly set to shuffle_orders=False: upstream does not
forward the top-level flag to these pooling modules. Leaving the default True
randomized eval serialization and broke both permutation and checkpoint parity.
After correction, actual BCE/FM permutation error is zero and checkpoint gradient
differences are around1e-10. Registered tolerances were not relaxed. Original
legacy models/vendor code are preserved. Failed attempts remain in experiments.
