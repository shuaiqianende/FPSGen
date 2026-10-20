# HS-OccFlow Full FP16 V6 pilot

## Result

A full explicit-FP16 inference copy runs successfully for the Stage1 encoder, Stage1 flow, and Stage2 PTv3 while preserving FP32 geometry, indices, source noise, and Euler integration. The best tested combination is explicit FP16 weights/major activations in all three components, with the existing compiled Stage1 Flow, existing GPU pipeline, and multilevel PTv3 topology cache.

| Variant | Encoder ms | Flow ×2 ms | Stage2 ×4 ms | Generation ms | Peak allocated GiB | Quality |
|---|---:|---:|---:|---:|---:|---|
| V0 AMP + compiled Flow, 3-run mean | 204.12 | 59.61 | 143.74 | 431.60 | 3.198 | Reference |
| Full explicit FP16 + compiled Flow, 2-run mean | 179.58 | 59.01 | 97.59 | 362.40 | 1.994 | 41-frame aggregate screen passed |

The candidate generation reduction is approximately 69.2 ms (16.0%); network CUDA event mean moved from about 407.47 to 336.17 ms. Peak allocated memory fell about 1.20 GiB. The results are two complete 5+20 candidate runs, paired in alternating order against three V0 runs. Candidate round 3 is `NOT RUN` because the raw input directory could not be recovered from the current execution context; no output was synthesized to fill that slot. The first-round and second-round candidate means were 361.94 and 362.86 ms, respectively, showing close repeatability across the two completed runs.

## Answers to the experiment questions

1. **Explicit FP16 versus AMP:** yes, the full-network explicit variant materially improved this pilot. It also halved model parameter storage. The run supports lower cast/activation overhead and faster half kernels as plausible contributors, but the pilot did not isolate cast-kernel counts with a dedicated kernel trace, so that causal detail remains unproven.
2. **Encoder:** mean fell from 204.1 to 179.6 ms, about 12.0%; this is the measured best result in the completed full benchmark.
3. **Stage1 Flow:** no reliable gain. It stayed near 59 ms because AMP+Compile was already optimized. Explicit FP16 compiled Flow worked but was 60.36 ms in a one-frame smoke.
4. **PTv3:** Stage2 four-step mean fell about 32%, from 143.7 to 97.6 ms in the full candidate runs.
5. **FP32 islands:** geometry/indexing, frequency/time math where needed, and Euler state integration remain FP32. Full mode runs normalization modules in FP16. Selective normalization was slower in the encoder smoke; Stage2 selective mode was not formally benchmarked.
6. **Conversions:** model parameters and representative Conv/Linear/SparseConv activations are FP16 in the explicit run. Geometry remains original precision. A dedicated cast-kernel count was not collected.
7. **Memory:** allocated peak fell 37.6%, reserved peak 46.2% in these runs.
8. **Stable operation:** full explicit mode completed 2 × 20 timed frames plus the 41-frame quality prediction run, with no NaN/Inf or dtype failure.
9. **Quality:** all 41-frame aggregate screening limits passed; see the quality report for exact metrics and worst-frame deltas. Not exact parity.
10. **Below 400 ms:** yes, 361.94 and 362.86 ms means were measured for full generation, with jitter excluded.

## Limitations

The registered three-round candidate benchmark is incomplete, explicit-FP16 Eager Flow was not measured, the full 1.08M-point jitter path was skipped, and 408-frame quality was not run. Consequently this is a validated promising pilot, not a finalized default precision policy. Do not promote it as the repository default until the third candidate timing round and any required longer evaluation are completed.

Raw benchmark output remains under the ignored local `experiments/hs_inference_full_fp16_v6/` directory and is not committed. The tracked source, tests, and reports contain no checkpoints or dense prediction artifacts.
