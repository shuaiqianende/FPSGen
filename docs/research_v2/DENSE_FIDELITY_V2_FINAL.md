# HS-OccFlow Dense Fidelity V2 — Current Status

## Result

The V2 branch is created from the requested OMD V1 commit and contains the
failure audit, three isolated dense Stage0 candidates, registered 20K
configs, and CPU tests. GPU access is now confirmed through the host execution
context. The study is **IN PROGRESS**: real-shape speed screens and all three
distributed smoke runs passed; sequential registered 20K training has started.
No new trained quality claim is available yet.

## R1/R2 evidence

The 900-frame saved curves show that R1 Bottleneck-64 loses ranking quality:
at the same K=84,963 its IoU is 0.28637 versus R0's 0.37135, and precision is
0.58024 versus 0.70837. At threshold 0.5, R1 predicts about 25K more parents
per frame while macro precision falls by about 19 percentage points. R2 is
weaker still (fixed-K IoU 0.22953).

The static R2 EMA audit finds `context_scale` magnitudes no larger than 0.00560
and gate-bias sigmoid values between 0.475 and 0.525. This rules out only a
large scalar gain or strong bias-only axis preference. It does not isolate the
context branch's causal effect; actual branch activation norms and a
same-checkpoint context ablation remain scheduled GPU diagnostics.

## Candidate status

| Variant | Code/config | CPU shape + backward | GPU timing | 20K training | 900-frame quality | TensorRT FP16 |
|---|---|---|---|---|---|---|
| R0 Original | available reference | prior V1 | historical only | prior V1 | prior V1 | not part of this run |
| E1 Bottleneck-96 | implemented | PASS | 170.35 ms encoder screen | IN PROGRESS (438/5,000 updates at last check) | NOT RUN | NOT RUN |
| E2 Asymmetric 3D rank-64 | implemented | PASS | 157.11 ms encoder screen | QUEUED | NOT RUN | NOT RUN |
| E3 Lightweight Skip | implemented | PASS | 165.53 ms encoder screen | QUEUED | NOT RUN | NOT RUN |

CPU tests confirm the candidate blocks preserve tested feature shapes and
backpropagate finite gradients. Real Encoder outputs retain the registered
three pyramid shapes and are finite. All three GPU2+GPU3 two-update smokes
reported finite gradients and synchronized parameters. Initial random-weight
full Encoder screens show 4.0%/11.4%/6.7% speed gains for E1/E2/E3 over same-run
R0; this is only a training-entry speed screen. All three registered configs
retain the fixed 20K exposure/5K update and Stage1 geometry/NFE contract.

## Remaining gates

The full 20K pilots have not completed. The E1 training job is running on GPU2+GPU3;
E2 and E3 are queued in the same sequential job. At the observed rate of about
2 seconds/update, this is a multi-hour queue. After each finishes, evaluate its
EMA checkpoint on the frozen 900-frame manifest and fixed K values
`78,694 / 84,963 / 94,087`. Only candidates that preserve the registered
quality thresholds proceed to same-model PyTorch FP16 versus TensorRT FP16
comparison. The R2 dynamic branch audit is still pending. 100K training remains
outside this study's automatic scope.

The fidelity-specific TensorRT FP16 export, engine-build, and full-Encoder
benchmark scripts are now prepared and pushed. They have passed Python compile
and CLI smoke checks; actual engine export/build and timing remain NOT RUN until
a trained EMA candidate passes the 900-frame quality gate.

E1's registered 12-frame light validation at 2,000 exposures completed and was
finite, but its threshold-0.5 IoU was only 0.03545 with 1.57M active parents
per frame. This is an early training-health observation, not a candidate
quality result or a stopping decision; the registered comparison remains the
full 20K exposure and frozen 900-frame evaluation.
