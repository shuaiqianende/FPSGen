# Direct Source-to-GT Sparse Sinkhorn: B10 Diagnostic

## Scope

This is an offline diagnostic only. It does not train or modify the Teacher or
Student. Each of the ten sequence-08 frames reuses exactly the cached FPSGen
source cloud `P0`, Teacher endpoint, and GT from
`outputs/research_v2/sinkhorn_cache/b10/`.

The question is whether the Teacher can be removed when constructing a
training target:

`P0 -> Sparse Sinkhorn(P0, GT) -> P_direct`.

All direct runs use `epsilon=0.01`, `iterations=100`, and `alpha=1.0`.
The Teacher+Sinkhorn reference remains `K=16`.

## Metrics

`exact_duplicate_rows` is the number of surplus identical XYZ rows: a group
`[a, a, a]` contributes two. Self-NN uses the second neighbour (`k=2`), so it
does not count each point itself. The three threshold columns are ratios of
points whose true self-NN is below the stated distance.

## Geometry and K Ablation (10-frame mean)

| Method | K | Chamfer | F-score | Coverage | NN target coverage | Runtime (ms/frame) | Peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw P0 | - | 0.57567 | 0.25318 | 0.33148 | 0.35296 | - | - |
| Raw Teacher | - | 0.13811 | 0.86562 | 0.89127 | 0.63106 | - | - |
| Direct Sinkhorn | 16 | 0.07197 | 0.95723 | 0.94178 | 0.66918 | 467 | 519 |
| Direct Sinkhorn | 32 | 0.07250 | 0.96451 | 0.95677 | 0.69182 | 912 | 1026 |
| Direct Sinkhorn | 64 | 0.07280 | 0.96627 | 0.96079 | 0.70016 | 2291 | 2040 |
| Teacher + Sinkhorn | 16 | 0.06862 | 0.98602 | 0.98732 | 0.76080 | - | - |

Larger direct `K` improves coverage slightly, but worsens Chamfer and raises
cost. More importantly, it does not resolve the many-to-one distribution
problem below.

## Point-distribution comparison

Two B10 GT frames (`000904`, `001357`) already contain 66k+ exact duplicate
rows in the cached GT. They cannot serve as a clean test of newly introduced
clustering. The main comparison therefore uses the other eight frames whose
GT has zero exact duplicate rows.

| Method, 8 clean-GT frames | Exact duplicate rows/frame | NN <1 cm | NN <2 cm | NN <5 cm | NN p01 (m) | NN p05 (m) | NN p50 (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT | 0.0 | 0.002% | 0.039% | 1.829% | 0.04425 | 0.06745 | 0.14288 |
| Raw Teacher | 0.0 | 0.016% | 0.090% | 0.991% | 0.05194 | 0.08976 | 0.17567 |
| Direct Sinkhorn, K=16 | 692.0 | 6.258% | 9.361% | 19.714% | 0.00002 | 0.00697 | 0.11876 |
| Teacher + Sinkhorn | 10.4 | 0.320% | 1.062% | 6.879% | 0.01951 | 0.04407 | 0.13232 |

Across all ten frames, direct K=16 has 852.1 exact duplicate rows/frame and
8.870% / 12.586% / 23.956% below 1 / 2 / 5 cm. K=32 reduces those to 621.2
and 5.865% / 8.870% / 19.528%; K=64 reduces them to 585.4 and 5.078% /
7.789% / 17.971%. The reduction is real but remains far from the GT and
Teacher+Sinkhorn distributions.

## Interpretation and decision

Direct Sinkhorn obtains an attractive Chamfer score because multiple `P0`
rows frequently map to the same or nearly the same local barycentre. This is
not a valid replacement for the Teacher target: it has strong many-to-one
collapse in clean-GT frames. The Teacher substantially suppresses this effect
before Sinkhorn refinement.

**Decision: HOLD / do not start the O2 Direct-OT Student experiment.**

The diagnostic script writes per-frame values to ignored runtime outputs:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_direct_source_sinkhorn.py \
  --cache outputs/research_v2/sinkhorn_cache/b10 \
  --output outputs/research_v2/direct_source_sinkhorn/b10_k16_distribution \
  --k 16 --epsilon 0.01 --iterations 100 --visuals
```

The saved four-frame visual PLY comparison is under:

`outputs/research_v2/direct_source_sinkhorn/b10_k16_distribution/visuals/`.
