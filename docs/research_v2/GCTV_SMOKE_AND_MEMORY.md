# GCTV engineering status

The original fp32-index prototype passed real maximum-parent Batch2 forward/backward on RTX3090, then was corrected to use the original f64 voxel indexing. G0/G1/G2 each passed final-source real-Batch2 checks, with allocated peaks approximately 4.52/4.61/4.69 GiB, respectively. These are engineering checks, not steady-state benchmarks. Gate JSONs bind configuration and source hashes. The final G1/G2 code also preserves identical same-seed shared initialization.

9 CPU tests passed: empty/single input, duplicate/voxel order, actual f64 boundary example, non-square XYZ projections, lifted axes, feedback-off equivalence, output/gradient dependence on feedback after projections open, small-volume output shapes/gradient, and same-seed common G1/G2 parameters. The three largest real Batch2 checks record full scene IDs and AMP. Formal native 32-successful-update smoke, optimizer/RNG continuation replay, FP32-versus-AMP error, per-module kernel profiling and warm full-encoder performance are NOT TESTED.

Resource admission blocked one G0 attempt while another compute process was present; the later attempt ran only after the card was idle. One G1 launch used GPU3 with a GPU2-bound config and correctly failed its device identity check; it did not run a model. No other processes were terminated.
