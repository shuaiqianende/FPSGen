> 当前精度方案已更新：用户选择显式FP16网络＋FP32主权重，旧运行已保存停止。最新代码、实际运行和接续队列见[显式FP16训练报告](DESESCENE_EXPLICIT_FP16_TRAINING_V1.md)。下文仅为旧精度启动历史，不代表当前任务。

> 后续三组已接入实际自动队列，见[接续计划与控制器状态](DESESCENE_SCRATCH20K_FOLLOWUP_QUEUE.md)。现有两张卡训练保持不动，以下为此前启动快照。

# 从零20K已正式启动

2026-10-26服务器核验：GPU2 Stage1 MeanFlow，完整Encoder+Flow均随机初始化并共同训练；GPU3 Stage2普通FM，GT粗占用+原始Observed Mask，无Encoder/384D特征，PTv3全随机初始化。两组Batch2、Seed20261032、总20K成功曝光/10K更新。

两组均已通过自己的最大真实Batch2、梯度、完整状态磁盘恢复、原生重放固定界限和独立32步Smoke，正式启动均resume=None、source_exposures=0、pretrained_checkpoint_loaded=false，初始SHA与各自随机初始化工程Gate一致，不复用Smoke训练权重。GPU2为满足原重放界限采用FP32 Encoder精度岛、Flow AMP；GPU3原FP32参数+FP16 AMP。

正式进程GPU2 PID1522599、GPU3 PID1495624；实际计数见快照`scratch20k_v1/both_gpus_formal_snapshot.json`或服务器train.jsonl，不从PID推断更新数。GPU2独立修复分支research/desescene-scratch20k-stage1-deterministic-v1；GPU3分支research/desescene-scratch20k-v1。

训练结束自动调用seq08/skip100固定41帧、NFE1/2/4/8评价，模型和精确CD均GPU。Stage1独立Parent中心结果与Stage2 Oracle-Support Fine结果分别标记，不能宣称成熟部署系统质量，也不与旧额外20K适配混为Scratch对照。质量、正式推理速度、完整系统提升均NOT TESTED。

18项CPU测试含真实模型构造禁止torch.load，均通过。GPU失败试验和所有旧微调产物保留。没有新增100K、第二Seed、Token截断或其他架构。
