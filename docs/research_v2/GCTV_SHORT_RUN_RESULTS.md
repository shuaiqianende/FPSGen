# GCTV staged training status

The baseline seq08/skip100 41-frame diagnostic completed for V1, V2 and original scratch20K FM. At NFE2, Parent IoU is 0.319459 / 0.325237 / 0.388743, respectively. This is not a strictly matched training-protocol ablation. The 101-threshold PR curve integration is an approximate diagnostic; the reference was reevaluated with FP32 weights + AMP rather than its original explicit-FP16 inference policy.

G0 on GPU3 and G1 on GPU2 passed individual 32-successful-update smokes and started independent scratch training. G2 is queued behind G1 on GPU2. Each has a 20K exposure scheduler horizon; at 2K exposures it stops for seq08/skip100 41-frame evaluation. A separate supervised queue resumes each healthy 2K run to 5K exposures with its own optimizer/scheduler/RNG checkpoint, and evaluates the same 41 frames. No automatic 20K extension is registered. 2K/5K quality, paired deltas and final selection remain NOT TESTED until actual results exist.

Only seq08/skip100 41-frame diagnostics/evaluations are authorized after the user's correction. The older 900-frame diagnostics were stopped and partial records preserved. Training uses the original train manifest; that fact does not authorize running the old 900-frame evaluation.
