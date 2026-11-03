# Stage2 real Batch 4 paired scratch experiments

GPU2 trains Stage2 with the frozen T1 depth-moment U1+A LiDAR encoder; GPU3 trains Stage2 without a LiDAR encoder branch. Both remove the Stage1 score branch, use GT parent support without neighboring negative parents, raw-input observed masks and the registered morphology labels. The encoder arm does not run Stage1 flow. Neither Stage2 model is initialized from a trained Stage2 checkpoint.

Both use the complete 00–07/09/10 training manifest, seed 20261032, real single-GPU batch 4, 25,000 successful optimizer updates and 100,000 successful frame exposures. Network weights are FP16, master parameters, Adam state and EMA FP32. The fixed-shape time MLP uses Inductor; PTv3 is not compiled. The final seq08/skip100 41-frame evaluation remains to be implemented for these model interfaces; it is not implied by training completion.

Each independent tmux queue first runs an untruncated maximum-scene Batch 4 forward/backward, then a separate 32-successful-update smoke. Formal scratch training starts only after both pass with matching source and config hashes. A failed memory or numerical check stops its arm rather than reducing batch or scene size. Run identity, AMP skips, sample IDs, update count, timing and memory are recorded in the experiment directory. A saved formal checkpoint can be resumed explicitly with `--mode formal --resume` after checking its identity.

Launch: `bash scripts/research_v2/hier_occ_flow/run_stage2_b4_paired.sh with_encoder` on GPU2 and the same command with `no_encoder` on GPU3; the script sets CUDA visibility. Existing earlier experiments remain separate.
