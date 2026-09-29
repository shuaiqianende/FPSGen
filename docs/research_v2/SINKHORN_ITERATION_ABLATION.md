# Sparse Sinkhorn iteration ablation: CD+rep versus DCD-only Teacher

## Scope

This endpoint-only ablation changes **only** the number of Sparse Sinkhorn
iterations.  It compares the historical CD+rep Teacher and the DCD-only
Teacher under the identical protocol:

```text
evaluation split : fixed, non-adjacent SemanticKITTI seq08 B10
frames           : 000000, 000452, 000904, 001357, 001809,
                   002261, 002713, 003166, 003618, 004070
GT               : gt_possion (0.15 m fixed-voxel + viewpoint-supported)
Sinkhorn K       : 8
epsilon          : 0.002
Sinkhorn alpha   : 1.0
iterations       : 25, 50, 100, 200, 500
```

For each Teacher separately, the ten cached `P0` and GT pairs are reused for
every iteration count.  The evaluator verifies them bit-identical before
computing metrics.  Thus, a row differs from another row only in the Sinkhorn
iteration count; this is not a re-sampled-source comparison.

The DCD cache is from the epoch-03 checkpoint and is located at
`outputs/research_v2/dcd_teacher/cache_b10_epoch03_gtpossion`; the matching
CD+rep cache is `outputs/research_v2/dcd_teacher/cache_b10_cdrep_gtpossion`.
The results are selection diagnostics, not a replacement for the pending
final-checkpoint B100 robustness evaluation.

## Geometry and coverage

Runtime is isolated Sinkhorn refinement time per frame.  It excludes endpoint
geometry and density metric calculation.  Lower Chamfer is better; higher
F-score, coverage, and NN-target coverage are better.

| Iterations | DCD CD ↓ | CD+rep CD ↓ | DCD F-score ↑ | CD+rep F-score ↑ | DCD coverage ↑ | CD+rep coverage ↑ | DCD target coverage ↑ | CD+rep target coverage ↑ | DCD runtime | CD+rep runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25  | 0.051255 | **0.050893** | **0.988064** | 0.985652 | **0.981064** | 0.977633 | **0.867745** | 0.855062 | 123.5 ms | 123.2 ms |
| 50  | **0.052346** | 0.052488 | **0.989365** | 0.986504 | **0.984064** | 0.979847 | **0.882399** | 0.867702 | 136.8 ms | 135.8 ms |
| 100 | **0.052840** | 0.053370 | **0.990150** | 0.987064 | **0.986229** | 0.981593 | **0.893678** | 0.878427 | 164.1 ms | 161.2 ms |
| 200 | **0.052735** | 0.053544 | **0.990956** | 0.987462 | **0.988255** | 0.982938 | **0.903794** | 0.887924 | 218.1 ms | 215.4 ms |
| 500 | **0.051936** | 0.052949 | **0.991864** | 0.988112 | **0.990295** | 0.984681 | **0.914342** | 0.899356 | 380.8 ms | 374.7 ms |

The apparently favorable CD+rep Chamfer at 25 iterations must be read with
the density diagnostics below: it is accompanied by much stronger many-to-one
accumulation, so it is not a preferred operating point.

## Local clustering / duplicate diagnostics

Values are averages over the same ten frames.  Close-neighbour quantities are
the percentage of output points whose true self-NN is below the given distance.
Lower is better.

| Iterations | DCD exact duplicates / frame ↓ | CD+rep exact duplicates / frame ↓ | DCD NN<1 cm ↓ | CD+rep NN<1 cm ↓ | DCD NN<2 cm ↓ | CD+rep NN<2 cm ↓ | DCD NN<5 cm ↓ | CD+rep NN<5 cm ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25  | **493.6** | 826.1 | **2.671%** | 4.423% | **3.406%** | 5.357% | **6.213%** | 8.207% |
| 50  | **269.8** | 525.3 | **1.210%** | 2.518% | **1.642%** | 3.123% | **3.707%** | 5.356% |
| 100 | **132.9** | 323.4 | **0.549%** | 1.466% | **0.830%** | 1.874% | **2.486%** | 3.668% |
| 200 | **64.8** | 205.7 | **0.260%** | 0.866% | **0.476%** | 1.153% | **1.938%** | 2.629% |
| 500 | **20.6** | 108.9 | **0.129%** | 0.463% | **0.315%** | 0.670% | **1.701%** | 1.883% |

## Interpretation and fixed target choice

1. More iterations improve coverage and reduce local point accumulation for
   both Teachers.  This confirms that 25 iterations produces an insufficiently
   settled local reweighting, despite its low CD in one row.
2. DCD-only is better than CD+rep at every matched iteration from 50 through
   500 on geometry, coverage, exact duplicates, and all three close-NN rates.
   At 25 iterations, CD+rep has a marginally lower Chamfer but DCD still has
   materially lower clustering.
3. The endpoint-only Pareto curve continues to improve through 500 iterations,
   but runtime rises from about 164 ms/frame at 100 to 381 ms/frame at 500.
4. The Student target remains intentionally fixed to the requested pragmatic
   configuration:

   ```text
   DCD-only Teacher + Sparse Sinkhorn
   K=8, epsilon=0.002, iterations=100, alpha=1.0
   ```

   It is roughly 2.3x cheaper than 500 iterations while already improving
   substantially over CD+rep at the same 100-iteration budget.  The 200/500
   rows are preserved as endpoint ablations; they do not silently change the
   Student training protocol.

## Reproducibility artifacts

```text
outputs/research_v2/dcd_teacher/b10_dcd_e03_iteration_sweep/
outputs/research_v2/dcd_teacher/b10_cdrep_iteration_sweep/
```

Each directory has one `summary.json` per iteration count, including the
per-method metric aggregate, runtime, peak allocated memory, and the verified
`P0` identity assertion.  Generated caches and JSON output are intentionally
not committed to Git.
