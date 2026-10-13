# SS-VSF environment and resource preflight

所有本路线CUDA作业限制 `CUDA_VISIBLE_DEVICES=2`、`3` 或 `2,3`。launcher在CUDA调用前查询nvidia-smi；双卡内部logical0/1对应physical2/3，不硬编码cuda:2/3。没有停止其他作业。

独立环境绝对路径：`/data-12/M2024-HWZ/FPSGen/env/voxdet_voxel_shortcut`。从已经可运行的VoxDet环境复制独立site-packages建立，无hardlink共享可写文件，launcher指向新环境。系统Python、CUDA、driver、bashrc和现有FPSGen环境均未修改。

| 项目 | 实际环境 |
|---|---|
| Python | 3.9.19 |
| PyTorch | 2.0.1+cu117 |
| MMCV-full | 1.7.2 |
| CUDA toolkit | 11.7；系统原配置保留 |
| Driver | 580.159.03 |
| GPU | RTX3090，24GB；只用physical2/3 |

相对官方Python3.7/PyTorch1.10.1/cu113的兼容差异：复用已审计的standalone registry适配，不安装MMDetection/MMDetection3D/SpConv；当前官方dense LiDAR执行路径不需要这些框架依赖。没有以更小3D U-Net替换VoxDet主干。网络资源不足时允许修改的新velocity U-Net与保留的VoxDet是不同模块。

Torch extensions、KeOps、XDG和pip cache均由launcher设在 `env/voxel_shortcut_cache`，KeOps按visible GPUs/rank隔离。没有修改全局环境配置。AMP初始scale4096；log1p、softplus、loss reductions和最终decode为FP32。学生checkpointing启用，EMA eval/no-grad。优化器只在全卡梯度finite时成功更新，AMP跳步单独记录。

Base独立65参数occupancy head已验证GPU2 FP32、GPU3 FP16、GPU2+3 DDP及64步双卡pilot。CFM和Shortcut均已通过单卡FP32、FP16和DDP4步 smoke，完整状态 `[1,1,32,256,256]`。实际显存和吞吐由同目录 `ss_vsf_reports/resource_summary.json` 汇总，包含cold-start和steady-step测量，不把pilot小型velocity配置称作显存/吞吐最优。

当前 B1/GPU launcher会拒绝配置其他batch以避免名义batch与实际实现不符。下一轮可在独立实验中测更大velocity容量或真正batched场景处理；不得通过缩小正式体素尺寸获得吞吐。

最终新增BatchNorm checkpoint buffer隔离修复，并完成CPU普通/checkpoint输出/梯度/stat parity及GPU2+3 DDP4步逐buffer更新检查。已有冻结pilot使用修复前的训练buffers；运行时eval不会重算，所有对照的frozen base一致。详细限制在结果报告中明确记录。

正式长训当前未启动。必须先冻结训练预算、capacity、source、gate和loss协议，并通过阶段gate。报告中记录依赖/数值correctness通过与模型质量尚未验证之间的区别。
