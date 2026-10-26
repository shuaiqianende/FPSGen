# 从零20K：Stage1 MeanFlow / Stage2无Encoder普通FM

用户2026-10-26明确要求GPU2/GPU3分别从零训练。两组独立模型随机初始化，不调用任何预训练加载器，不复用旧微调/Smoke的权重或曝光。20,000成功帧曝光、Batch2、10,000成功更新，Seed20261032。所有后续质量仅seq08/skip100固定41帧，NFE1/2/4/8。

- S1：原始Dense Encoder＋原宽度8/16/32/64 Flow，全体可训练，MeanFlow原生JVP/75%对角混合，FP32噪声与积分，FP16 AMP。Encoder LR1e-4，Flow LR2e-4，恢复Scratch学习率，不沿用微调1e-5。
- S2：原动态PTv3直接随机构造condition_mode=observed，原64D宽度/8D状态和速度头，fuse176D；无384D特征、无Encoder/Stage1 Flow，无interval/JVP，原高斯到二值FM。GT coarse＋25%邻近空Parent，Score均1，Partial LiDAR Observed Mask。Stage2 LR2e-4。
- AdamW wd0.01、500成功更新预热后Cosine、Clip1、EMA0.999、AMP初始scale2048。仅用于FP16数值管理，不改模型/几何/输入/目标。
- 数据仍为原17870训练Manifest，不加入seq08。900 Development仅可用于工程最大场景索引核验，不再跑900质量评价。

独立训练器：`fpsgen/hier_occ_flow/scratch20k_training.py`；流水线：`scripts/research_v2/scratch20k/run.py`。每组先验收完整最大Batch2、梯度、参数更新、Checkpoint/EMA/Adam/Scheduler/Scaler/RNG精确磁盘恢复与原生下一步的固定界限，再32步丢弃Smoke，正式训练重新初始化为同一随机初态和计数0。不中途缩Token/降Batch，失败即停，不自动放宽数值限值或重启。

CPU16项通过（Scratch构造不调用权重加载器、所有Encoder/Flow可训练、可微条件穿过MeanFlow目标计算、普通FM无interval、Resume不可伪装微调、原MeanFlow数学回归等）。

首次工程结果：S2 GPU3最大399672+399671=799343 Parents通过；S1 GPU2完整384³XY/32Z真Batch2，原生JVP与Encoder反向跑通，内存约9.98GiB，但下一更新重放22个参数/Adam项超既定容差，故BLOCKED而非通过。该失败报告全部保留。另立S1确定性执行修复，不能修改已进入Smoke的S2源码身份。

评价脚本`experiment_ops/eval_scratch20k_seq08.py`只加载本次最终Scratch EMA，不加载成熟权重。Stage1先报告独立粗占用中心质量；Stage2报告Oracle支持下原生Fine中心质量（非部署）。这两种质量不能混称为完整系统指标，也不能直接与旧微调表作曝光匹配因果比较。完整两阶段组合需两组完成后另行明确支持身份。

训练日志/大Checkpoint在experiments本地，不上传Git。新旧实验分支、运行目录严格分离。新质量和速度尚未完成，不预报性能收益。

## Stage1重放修复：独立分支GPU2

首次原AMP Gate以及仅启用确定性cuDNN的重试都因少量参数更新差异超既定容差失败，产物均保留，未变更bounds或忽略异常。最终采用Encoder FP32精度岛、Flow FP16 AMP，关闭TF32、启用确定性cuDNN，在同样的完整最大Batch2上通过原生下一更新重放。参数、架构、几何与曝光均不改变；这属于明确登记的计算精度策略，不能把它说成全网络AMP。后续同协议对照必须匹配该精度。

最大场景为seq02帧925/4192、GT positives319738/319737；三尺度输出[2,128,16,192,192]/[2,128,8,96,96]/[2,128,4,48,48]。Encoder梯度范数0.089326、Flow7.489892，原生JVP实际调用1次，initial state SHA1e7837680e8100176761627264c188a1d0d351cc8a65dceab6038507257f4058。梯度重放relativeL2=3.6851e-6，原界限PASS；磁盘全状态精确恢复，归类NUMERICAL_NATIVE_REPLAY_WITHIN_FIXED_BOUNDS，不宣称逐位更新相同。峰值allocated12.072GiB。

GPU2实际接续路径为`experiments/desescene_scratch20k_encoder_fp32_v1/S1-MeanFlow-Scratch20K`，controller1500737；GPU3在原独立工作树`desescene-scratch20k-v1`中，controller1484049，两者不共享可变源文件。32步Smoke丢弃，正式仍从计数0且全随机初值开始。
