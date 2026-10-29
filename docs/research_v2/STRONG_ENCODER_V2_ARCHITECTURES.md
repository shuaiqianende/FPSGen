# Strong Encoder V2 — registered first phase

The original 6D point MLP, both 3D ResNets, both FPNs, and classification TPV remain intact. S1 returns the second backbone's F1/F2 neck outputs and F3 backbone output. S5 leaves the encoder untouched and injects an eight-channel, zero-initialized projection of F1 after the last flow decoder block. Its inference context caches that low-resolution projection; interpolation and Euler integration remain per step.

S1/S5 are the only variants currently implemented. S2/S3/S4/S6 and the six-arm ranking/20K continuation are NOT IMPLEMENTED and NOT RUN. Existing H/G histories are unchanged. The first-phase runner stops at 5K after a separate non-seq08 development screen; no 900-frame evaluation, automatic 20K continuation, or Stage2 training is performed by it. This pause is intentional because selection of 1–2 candidates per GPU requires the other registered arms' 5K results.

GCTV 20K seq08/41 NFE2: G0 IoU .368868/CD .308363; G1 .374413/.309614; G2 .375480/.301810. H1 .371375/.304220; H2 .372992/.302095; H3 .374920/.302356; H4 .374729/.295342; H5 .372471/.305816. Matched original VoxDet .389428/.265783. These numbers measure quality only; no unmeasured runtime gain is inferred.
