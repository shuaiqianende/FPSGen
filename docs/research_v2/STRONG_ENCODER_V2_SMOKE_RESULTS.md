# Strong Encoder V2 — initial checks

Both S1 and S5 passed full real maximum-scene Batch 2 AMP forward/backward and checkpoint/EMA roundtrip. S1 peak allocated 11,269,251,072 bytes; S5 11,184,063,488 bytes. Their initial unoptimized cold event durations are not formal inference latency measurements. Both preserve output shapes `[B,128,16,192,192]`, `[B,128,8,96,96]`, `[B,128,4,48,48]`.

S5 zero-residual equality with the original Flow and projection gradients were tested on a small CPU network, including inference-context parity. The registered 32-update GPU smoke is a separate prerequisite and its result must be read from the experiment status, not assumed from the engineering gate.
