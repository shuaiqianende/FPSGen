# HS-OccFlow V5 Conv3d–BatchNorm Fusion

This was a one-frame GPU2 experiment on real Seq08 frame 100 with the V4 frozen checkpoint and FP16 autocast. The inference-only candidate operated on a copied model; the checkpoint and training model were not modified.

The script identified 33 eligible `Conv3d → BatchNorm3d → activation` sequences. Fusing all 33 increased encoder event time from 204.22 to 209.62 ms. The first pyramid output had relative L2 error 0.368; the next two had relative L2 errors 1.174 and 1.333 (maximum absolute errors 20.56 and 9.73). Voxel-backbone-only fusion of 15 layers also slowed from 203.94 to 206.67 ms and produced large downstream feature errors. A single isolated Conv/BN pair was close, but accumulated end-to-end behavior was not.

The multi-layer fusion is therefore **REJECTED**: it was slower and failed the feature-parity screen. No full generation or 41-frame quality run was warranted. The likely issue is that folding FP32 running statistics into weights interacts with the original autocast/rounding placement; the observed evidence is sufficient to reject this implementation, not to claim that every possible fusion implementation is mathematically invalid.

Raw evidence: `experiments/hs_inference_opt_v5/conv_bn_fusion/gpu2_frame100_strict.json` and `gpu2_voxel_backbone.json`. The original model remains unchanged.
