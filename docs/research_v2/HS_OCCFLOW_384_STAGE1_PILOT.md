#384 occupancy learning evidence

No density mapping/normalization, occupancy auxiliary, shortcut, teacher target,
Chamfer or mass loss is used. FM loss is masked velocity MSE only. BCE is plain
masked BCEWithLogits. Input XYZ and derived features/observed mask are the only
conditions. Seeds,8-frame training order, model widths and successful-update
budgets match between baselines. Development is the separate frozen8-frame
initial cohort; it does not replace full900-frame formal evaluation.

| Model | Scope | Updates | NFE | Macro F1 | Macro IoU | Recall |
|---|---|---:|---:|---:|---:|---:|
| fm | overfit8_pre_f64 | 200 | 1 | 0.1535 | 0.0834 | 0.3328 |
| fm | overfit8_pre_f64 | 200 | 2 | 0.1622 | 0.0886 | 0.3479 |
| fm | overfit8_pre_f64 | 200 | 4 | 0.1684 | 0.0923 | 0.3583 |
| fm | overfit8_pre_f64 | 200 | 8 | 0.1725 | 0.0948 | 0.3670 |
| fm | pilot_f64_dev8 | 1000 | 1 | 0.5437 | 0.3739 | 0.4688 |
| fm | pilot_f64_dev8 | 1000 | 2 | 0.5054 | 0.3386 | 0.4578 |
| fm | pilot_f64_dev8 | 1000 | 4 | 0.4822 | 0.3182 | 0.4643 |
| fm | pilot_f64_dev8 | 1000 | 8 | 0.4681 | 0.3060 | 0.4630 |
| bce | overfit8_pre_f64 | 200 | 1 | 0.5770 | 0.4077 | 0.4708 |
| bce | pilot_f64_dev8 | 1000 | 1 | 0.5521 | 0.3818 | 0.4597 |

The200-update overfit runs used the pre-FP64-size helper and evaluated their
8 training frames with online weights. They show learning capacity, not held-out
quality. The fresh1K pilots use the final precision-audited geometry helper and
evaluate online weights on8 held-out frames, plus train-set diagnostics at1K.
Pilot parameters/optimizers are never reused by formal training. Short-pilot
EMA still lags; online and EMA evaluation choices are explicitly labeled.

BCE is currently substantially stronger in the200-step comparison. FM loss
reduction alone is not proof of good occupancy or an advantage over BCE.
No claim that FM wins is made. Formal gates require real learning/no collapse,
finite gradients, successful resume and complete audited full data.
