# HS-OccFlow V6 component results

GPU2, RTX 3090, PyTorch 2.0.1+cu117. Values below are measured, not estimates. Full-model timing used the registered 5 real warmup frames and 20 timed frames; compile cold-start is recorded in protocol and occurs during warmup.

## Isolated pilot results

| Variant | Encoder ms | Stage1 Flow ×2 ms | Stage2 ×4 ms | Full generation ms | Result |
|---|---:|---:|---:|---:|---|
| AMP baseline V0, 3 rounds | 204.24 / 203.54 / 204.60 | 59.67 / 59.43 / 59.73 | 143.59 / 143.53 / 144.10 | 430.66 / 431.02 / 433.12 | Complete |
| Encoder explicit FP16 full, 20-frame round 1 | 179.53 | 58.93 | 97.63 | 361.94 | Complete |
| Encoder selective FP32 norm, one timed frame smoke | 206.32 | 59.26 | 189.81 | 489.48 | Runs, slower; stop |
| Flow explicit FP16 + compile, one timed frame smoke | 204.41 | 60.36 | 188.70 | 483.33 | Runs; no measurable gain |
| Stage2 explicit FP16, one timed frame smoke | 204.14 | 59.75 | 126.05 | 417.90 | Runs; included in full variant |
| Full explicit FP16 + Stage1 Flow compile, 20-frame rounds 1–2 | 179.53 / 179.63 | 58.93 / 59.08 | 97.63 / 97.54 | 361.94 / 362.86 | Complete |

The encoder full-FP16 warm smoke was 179.03 ms for frame 100. A cold, un-warmed smoke took 2.62 s in the encoder and is not a steady-state performance result. The selective-normalization and isolated module smokes each used five real warmups followed by one timed frame. One-frame module comparisons are diagnostic only, not a 20-frame speed claim.

## Observations

The full explicit variant reduced the three-round V0 mean generation time (431.60 ms) to a two-round candidate mean of 362.40 ms, a paired reduction of about 69 ms (16.0%). Across components, encoder time fell about 24.6 ms (12.0%), Stage2 four-step time about 46.0 ms (32.0%), while the compiled Stage1 Flow changed by only about 0.5 ms. Thus the main measured gain comes from the explicit FP16 encoder and Stage2 execution; explicit FP16 adds no reliable benefit to the already compiled Stage1 flow.

Peak allocated memory was 3.198 GiB for V0 versus 1.994 GiB for the full explicit candidate (about 1.20 GiB / 37.6% lower); peak reserved was 3.979 versus 2.141 GiB. The parameter storage itself is halved as documented in the dtype audit. Formal timing skipped jitter, so no 1.08M-point delivery latency is reported.

The benchmark has two completed candidate rounds and three completed V0 rounds. Candidate round 3 remains `NOT RUN`: the current restricted shell no longer exposes the raw seq08 files used in the first two rounds, and cannot safely replay the exact 5+20 input manifest. The GPU itself was checked outside the restricted shell and GPU2 was idle. Therefore the observed two-round candidate result is promising and consistent, but the requested three-round candidate repetition is incomplete; the V6 speed result should be treated as a two-round pilot, not a final three-round claim.
