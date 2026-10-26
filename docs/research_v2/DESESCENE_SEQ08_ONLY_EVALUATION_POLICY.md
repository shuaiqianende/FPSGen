> 更新：用户已纠正为从零总20K，旧适配训练已停止；见[初始化更正](DESESCENE_SCRATCH20K_INITIALIZATION_CORRECTION.md)。下面为此前执行快照，自动适配续训已失效。

# 2026-10-26：用户修订为 seq08/skip100 固定41帧

本页覆盖历史队列中“先完成900帧才能续训”和“900 + 41双评价”的安排。历史报告仍保留原实验身份；不删除已完成结果，不把已中止评价写成完成。

## 当前执行

- GPU2：Stage2无Encoder **普通FM** 正式训练已启动，真实Batch2、Seed20261032，目标10,000 successful updates /20,000额外曝光。GT粗Parent + 原25%邻近空Parent，原始Partial LiDAR Observed Mask，无384D特征、无Encoder前向、无JVP、无拓扑预热额外前向。不是MeanFlow，也不是Shortcut。
- GPU3：Stage1 MeanFlow已完成10,000 successful updates /20,000曝光，切换为seq08/skip100固定41帧，与匹配普通FM对照，NFE1/2/4/8；原Stage2固定NFE4。
- 旧Stage1 Development900评价由用户明确要求跳过，停止于106个完整场景；保留原848条逐场景/NFE/模型记录及停止凭证。不是900帧最终结果。
- 新控制器分别为PID1121977（GPU2）和1166715（GPU3，补齐原始GT缓存后显式恢复）。PID及进度是本次交接快照，后续以服务器supervisor与日志为准。

## 唯一正式评价集

质量评价及新正式性能测试均使用seq08的0,100,...,4000，共41帧；不得用其他场景替代或漏帧后称为41帧。保留原始GT、ROI、物理网格IoU、CD、JSD、源噪声策略和模型身份。固定帧集成为用户指定的研究评价集，不再将其称为未经重复使用的独立测试集。

训练数据、原训练配置Hash、曝光预算、Seed与Checkpoint格式完全不变。旧训练配置中900帧字段仅作为原始身份留存；本次调度明确以`--split seq08_41`覆盖评价入口。新策略文件：`configs/research_v2/hier_occ_flow/seq08_only_evaluation_policy.json`。

## 自动接续

独立调度器：`scripts/research_v2/experiment_ops/run_seq08_only.py`。它位于模型/训练源码Hash之外，单独固定自身Hash；不修改已验收或正在运行的训练代码。失败即停，不自动重试或追加新候选。

GPU2按100→500→10,000成功更新分段保存/严格恢复，然后只运行41帧普通FM评价（Oracle支持与原冻结Stage1 Predicted Support分别报告）。正式训练从原成熟源重新初始化，丢弃Smoke32权重；当前不再等待Stage1质量结束。

GPU3先完成41帧MeanFlow/普通FM质量比较，再使用`benchmark_meanflow_seq08_41.py`在同一GPU独占情况下，对全部41帧、NFE1/2/4/8运行三轮交替配对测速；预热仍为0/800/1600/2400/3200。Profiler和质量指标不参与正式计时。完整生成包含原生中心解码，不能与含108万点Jitter交付耗时混淆。Stage2无Encoder组的新正式端到端Benchmark尚未实施，标记NOT TESTED。

## GPU使用与结果边界

模型前向和最昂贵的精确CD采用GPU（现有CUDA KD-tree）；物理网格IoU与稀疏JSD保留经过验证的CPU实现，不宣称指标已全部CUDA化。数据读取与文件/GT身份核验仍需CPU/磁盘；启动后未输出首帧时不算已完成一帧。

新41帧质量、NFE优势及正式速度结果均待实际产物；本页不推测最终结论。当前Stage2对照为成熟全Encoder FM、无额外20K微调，不是曝光匹配的纯架构因果对照。

交接、控制器源码SHA和实际进度快照见`seq08_only_20261026/handoff_snapshot.json`。所有大型Checkpoint、训练数据和编译缓存仅留服务器。

## 缺失标签缓存的显式修复

首次41帧评价在首帧预测之前发现历史参考目录没有`targets/`，因此停止，0帧完成。原参考协议、原始地图/位姿/标定/扫描/生成代码的Hash均可核对。使用`prepare_seq08_target_snapshot.py`从这些原始来源独立生成41帧Surface-hit标签；逐帧核对历史raw GT点数、输入Hash和Geometry，固定新标签Hash。它是本次新建的标签缓存，不能宣称历史已经保存过该标签文件。

缓存放独立实验目录；只在旧参考目录原本缺失的`targets`路径增加链接，不修改任何旧协议、结果、日志和Checkpoint。修复后重新评价，首次失败目录/日志/控制器凭证全部另存。模型/训练配置Hash保持不变。

全部41帧新标签缓存已核验通过（`VERIFIED41`）；原参考文件SHA保持不变，恢复凭证见`seq08_only_20261026/verified_target_recovery.json`，逐帧标签/输入/原始GT核验见`raw_gt_target_audit.json`。质量结果仍待正式评价完成。
