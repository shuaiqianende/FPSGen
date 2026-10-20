# PTv3 Static Topology Cache Results

## Implementation

The opt-in `serialization` cache stores per-frame coordinates, batch/offset,
sparse shape and PTv3 serialized codes/order/inverse/depth. Every Euler step
still builds a fresh `Point` with its current features and time embedding.
`multilevel` additionally caches serialized-attention padding maps and the
static clustering/index maps used by serialized pooling. Pooling feature
projection and segment reduction remain dynamic. No SpConv indice cache is
used because spconv's mutable index metadata lifecycle was not established as
safe for cross-forward sharing.

Flags default to the original path:

```text
--ptv3-topology-cache none|serialization|multilevel
```

## RTX 3090 GPU3, same 20 timed frames

| Variant | Stage2 4-step mean (ms) | Full generation mean (ms) | 1.08M jitter mean (ms) | Gen + jitter mean (ms) | Peak allocated (GiB) | Fine occupancy parity |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 301.93 | 633.66 | 405.95 | 1039.61 | 3.01 | reference |
| Serialization cache | 263.54 | 604.70 | 400.63 | 1005.34 | 3.01 | 20/20 exact |
| Multilevel cache | 258.47 | 599.07 | 402.89 | 1001.95 | 3.01 | 20/20 exact |

Serialization alone reduced Stage2 time by 38.4 ms (12.7%) and full generation
by 29.0 ms (4.6%). The multilevel result reduced Stage2 by 43.5 ms (14.4%) and
full generation by 34.6 ms (5.5%). The incremental gain over serialization
alone is about 5 ms and needs repeated runs before treating it as robust. Batch
1 fastpath alone also passed exact parity but only changed generation from
633.66 to 629.81 ms (0.6%), which is within scene/runtime variation and is not
a meaningful standalone win.

Peak allocation stayed at 3.01 GiB. This cache is limited to a single scene's
four NFE forwards and has no training-path effect. The raw per-frame records
are under `experiments/hs_inference_opt_v3/benchmarks/`.
