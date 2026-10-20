# HS-OccFlow PTv3 Baseline Profiling

## Protocol

Profiling used the registered 384+8-way Stage2 on RTX 3090, with the frozen
Stage1 condition, predicted parents and four Stage2 Euler forwards. Three real
Seq08 scenes cover 77,828, 125,081 and 169,841 parent tokens. These are
diagnostic runs, not the 5+20 latency benchmark. CUDA event intervals inside
nested modules are inclusive and must not be summed as if disjoint.

## Findings

| Parent tokens | Stage2 four-step wall (ms) | Serialization CUDA events (ms) | SpConv modules (ms) | Attention (ms) | Pooling (ms) | Unpooling (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 77,828 | 221.3 | 39.9 | 51.4 | 34.6 | 8.2 | 6.2 |
| 125,081 | 320.8 | 44.8 | 74.9 | 52.8 | 11.2 | 9.3 |
| 169,841 | 395.5 | 46.4 | 86.5 | 67.5 | 12.8 | 11.1 |

Each of the four forwards repeats top-level serialization. It consumes about
40–46 ms over four steps, leaving a plausible cache opportunity. Attention and
pooling metadata are smaller. SpConv computation grows with token count and is
not just index construction; reusing its mutable `indice_dict` was not attempted.

CPU submit-side measurements also show substantial Python scheduling work in
serialization and pooling, but these host intervals overlap asynchronous GPU
execution and are not additive with CUDA event time. Stage2 serialization,
pooling, attention and SpConv all matter; no single operator accounts for the
full four-step latency.

The detailed machine-readable profile is
`experiments/hs_inference_opt_v3/profile/baseline_ptv3_3scenes.json`.
Nsight profiling was not run. No cache reuses activations or dynamic features.
