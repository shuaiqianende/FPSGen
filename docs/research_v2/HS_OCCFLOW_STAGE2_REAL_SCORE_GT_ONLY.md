# Stage2 实际 Stage1 Score + GT Parents Only（2026-10-18）

独立分支 `research/hs-occflow-384-stage2-score-gtparents`，基于 `c9b75c382da42792f067642a28a3a9fc5dca611f`。旧 worktree、模型、标签、日志全部保留。

## 实现与边界

- 每个训练父体素输入冻结 Stage1 的实际生成终点 score，替换常数1。Stage1 EMA checkpoint SHA：`b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`，4 NFE、固定场景 seed。该 score 不是校准概率。
- 候选为全部 GT 非空父体素，额外空父体素比例设为0，不再展开附近空父体素。低于 Stage1 0.5 score 的 GT 父体素仍保留。父体素内部的空子体素继续监督为0。
- Stage2 架构仍为64D Sparse PTv3 + 8维 Local FM，2,677,128参数；FM-only、原384+8-way几何均不变。
- 推理仍必须由 Stage1 预测选择候选，不能将本轮 Oracle 支持当作部署条件。去掉空父体素负样本后，对 Stage1 假阳性父体素的抑制可能变弱，需要单独验证，尚未声称质量提升。

## 已实际执行

- CPU回归：84 passed、1 skipped（该次CPU命令未分配CUDA）；GPU体素化对照在此前本轮单独CUDA测试通过。
- GPU2、GPU3分别完成真实B2打包场景的前向、FM反向和AdamW更新。两场景父体素数137,999/319,738；两份最大完整场景合计639,476tokens，不截断。
- 最大B2峰值 allocated12.826GiB、reserved14.141GiB。实际Stage1 score与采样条件逐元素一致；父体素顺序、场景隔离、FP32/AMP打包对照通过。
- 这不是新协议的DDP恢复或质量pilot通过证据；新数据版本尚未启动正式长训练，不复用旧工程gate。

## 原始稀疏度

完整训练17,870帧、开发900帧。训练GT非空父体素平均3.8761/8个子格有点（48.45%），全部8格非空的父体素只占8.35%。粗占用不意味着整格实体填满。

真实训练00/000000的1,322,038个原始候选点可精确重建当前标签；随机保留90%点仍保持95.21%的细占用，而180K Poisson仅覆盖原细占用约26.80%。本任务使用原始地图标签，不使用Poisson作为主GT。

## 旧训练真实状态

旧正式训练已完成100,000 successful frame exposures，共25,120次成功optimizer更新、13次AMP跳步。最终checkpoint：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes/checkpoints/step025120.pt`，SHA256 `2da939616c63c9d108be15bcb2fecb4ba3286d198e6db982801ca2a0a4a2adf7`。它沿用旧常数score/25%空父体素合同，不能当作新协议模型。

## 配置与证据

新 `stage2_384_real_score_gt_only_{smoke,overfit}.yaml` 是原始GT版本；`..._morph_{smoke,overfit}.yaml` 是独立形态学标签版本。均使用独立输出和未伪造的新工程gate。完整小型证据见同级 `stage2_real_score_gt_only_audit/`。
