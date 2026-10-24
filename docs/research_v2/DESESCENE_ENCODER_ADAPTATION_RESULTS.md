# Encoder adaptation: current evidence

Status: **P0 RUNNING; new formal training NOT STARTED.** Snapshot `2026-10-24T23:32:21.843469+08:00`: L0(position)6347/10000 successful updates,12694 exposures,4 AMP skips; L1(position+attention)5713/10000,11426 exposures,3 AMP skips. Existing training/queue sources remain unchanged. Neither final900 nor64-frame diagnostic is complete; no L0/L1 quality win is claimed.

| Check | Result |
|---|---|
| Original Stage1/Stage2 checkpoint SHA256 | PASS, both exact |
| Strict real source model construction/loading, CPU | PASS for EF1/EF2/S2-E0/S2-E1/S2-E2 |
| Shared initial Encoder parameter SHA | `7cd298c4555030ee02bfa8c141dfded579c6045b7d491e976aff6b8083dd422b` |
| Shared initial Flow state SHA | `16b378481f4f96bbf881913573f69d805f955ca2825d5a66cf95e621655ea598` |
| CPU differentiable frozen-Flow/Stage2 surrogate, freeze/EMA, resume and legacy regression tests |52 PASS; GPU PTv3 is not implied |
| Real maximum-pair Batch2 GPU checks | NOT TESTED; wait for P0 |
|32-update GPU smoke | NOT TESTED; depends on maximum-pair PASS |
| EF1/EF2 formal20K | NOT STARTED |
| EF0/EF1/EF2 full900 and repeated timing | NOT TESTED |

Dedicated runner `encoder_adaptation_runtime.py` is independent from the scratch runner. It records loaded tensors, trainable/frozen parameter names, initial hashes, complete optimizer/scaler/scheduler/EMA/RNG and sampler state, attempts/skips versus successful exposures, per-batch scene identities and peak memory. Frozen weights and running statistics, including frozen EMA state, must remain exact. Trainable Encoder must have finite nonzero gradients and change parameters.

Source initialization is verified on real models, but GPU gradient/replay/memory checks have not yet run. No new architecture quality or speed result is claimed. Phase-completion artifacts will be published under `docs/research_v2/desescene_encoder_adaptation_v1/` after actual completion.

Original17870/900 input files are all accessible; complete shard hashes passed. Original raw-map GT for Development00/000500 was loaded(1,079,847 XYZ rows). The machine-readable preparation audit includes source/manifest hashes, strict CPU initialization counts and checkpoint/source identity checks. GPU2 has unrelated compute jobs; admission never stops them, and new phases can use GPU3 sequentially if needed.
