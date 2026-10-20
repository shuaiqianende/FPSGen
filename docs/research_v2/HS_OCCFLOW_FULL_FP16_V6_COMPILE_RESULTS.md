# HS-OccFlow V6 explicit FP16 and compile results

The tested compile target is the already validated Stage1 Flow (`torch.compile`, PyTorch 2.0.1+cu117, `no_grad` compatibility path). The explicit FP16 Flow was constructed from a fresh FP32 checkpoint-loaded module, converted, then compiled, as required. It completed a 5-warmup + one-frame smoke without dtype errors or non-finite output.

| Flow mode | Weight storage | Execution | Measured Flow ×2 |
|---|---|---|---:|
| Existing compiled V0 | FP32 + AMP | Compile | 59.43–59.73 ms across three 20-frame runs |
| Explicit FP16 | FP16 | Eager | Not tested in this environment/input session |
| Explicit FP16 | FP16 | Compile | 60.36 ms, one-frame smoke |

The explicit-FP16 compiled flow is functionally compatible but did not improve on the established AMP+Compile flow; the 0.6 ms difference is below what this single-frame comparison can establish. The full-network explicit variant retained the compiled Flow and measured 58.93/59.08 ms across two formal rounds. Compile cold start was about 11.8 s in round 2 (first warmup frame); it is excluded from steady-state latency and included in protocol metadata.

The compile wrapper reports two compiled graphs and one graph break involving a tensor boolean / comparison, consistent with the existing V4 implementation. No silent eager fallback was used. Encoder compilation, CUDA Graph, and a complete eager-versus-compiled explicit Flow matrix were not part of this V6 pilot. The measured answer is therefore: compile continues to benefit the FP32-parameter AMP Flow relative to the historical eager path; explicit FP16 weights do not show an additional Flow speedup.
