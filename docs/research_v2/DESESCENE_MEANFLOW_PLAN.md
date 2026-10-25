# DeSeScene MeanFlow — 独立短验证与自动续训

登记日期：2026-10-26。规格以 [用户执行合同](meanflow/CODEX_MEANFLOW_SPEC_ZH.md) 为准。工作树 `worktrees/desescene-meanflow-v1`，分支 `research/desescene-meanflow-v1`，基础提交 `b9ad22e7f77e0b1ca9d9bdf7db8479798f55ac70`。

## 本次范围

暂缓 Stage2 Loss 更新 Encoder 的 S2-E1/S2-E2，保留 EF1/EF2、GPU0 FM 对照及其原自动评价计划。原 Shortcut Models 方案仍是独立登记项目；MeanFlow 不使用 EMA 两小步自举，不替代或重命名 Shortcut。没有自动 100K、额外训练 Seed、GT 补候选或新 Encoder 架构。

| 方案 | 训练模块 | 冻结模块 | GPU 偏好 | 正式预算 |
|---|---|---|---|---|
| S1-MF20K | Stage1 Flow + interval embedding | 成熟 EMA Encoder；Stage2 不训练 | 2 | 真 Batch2，10K successful updates / 20K exposures |
| S2-MF20K | 完整 Dynamic PTv3 FM + interval embedding | 成熟原始 Stage1 FM / NFE2 / threshold0.5 | 3 | 同上 |

共同 Seed20261032。S1 LR1e-5，S2 LR2e-4；AdamW wd0.01、warmup500/cosine 到10K、clip1、EMA0.999、原 AMP。来自成熟模型的额外20K微调，不称为 scratch20K。Stage2 源已有99,520次成功曝光。

## 数学与接入

采用 noise0 → data1，`d=s-t`：

```
D = jvp(U, (xt,t,s), (v,1,0))
target = stopgrad(v + (s-t)*D)
```

固定终点，必须包含状态导数。使用当前 online 网络，不使用 EMA target；无 FD 替代、无缺失导数置零。新增128D/32D时间分支为 `time_embedding(t) + d*interval_mlp(fourier(d))`，仅最后投影零初始化。旧共享 Tensor 的名称/值保留，新增键显式审计。普通 Forward、准备好几何的 JVP 路径与 cached inference 均接受 `time_end`。

原 Gaussian source / uniform current-time 的抽样顺序不变。独立 interval generator，逐场景 Bernoulli0.75 取对角，不将 Batch2 的75%错误取整成50%。主实验 adaptive_power=0，场景等权 masked MSE；invalid state/velocity/JVP/target/loss 使用同一 mask。

S1 Frozen Encoder 放在 JVP 外只运行一次。S2 在 JVP 外准备排序/坐标/序列化/场景长度；每步重新计算可训练静态投影，Point 和动态激活不跨步复用。项目内受控适配 SceneNorm 的静态 split 长度，保留原统计公式；没有修改 Vendor。FlashAttention、SpConv、Pooling/Unpooling 仍真实参与完整模型 JVP。不支持时记录准确堆栈并置 `BLOCKED_JVP`。

## 自动执行链

队列配置：[desescene_meanflow_queue.yaml](../../configs/research_v2/hier_occ_flow/desescene_meanflow_queue.yaml)。控制器：[run_meanflow_queue.py](../../scripts/research_v2/hier_occ_flow/run_meanflow_queue.py)。

1. 等待实际 EF1、EF2 的20K训练和900帧评价、EF0的900帧评价，以及 S1-FM20K 对照20K和900帧评价完成。检查最终产物；单纯 supervisor 的“完成”字符串不够。
2. 资源准入：不停止任何进程，GPU2/3 每卡一个本项目任务。用户允许外部任务共享时仍要求至少16GiB空闲；新进程 allocator cap14GiB，保留2GiB余量。真实最大 Batch2 不适配时停止，不偷偷缩小 Batch/Parent/场景。
3. 实际 native JVP、源输出零初始化、缓存/排序/双场景、最大真实 Batch2 AMP/反向、冻结状态及完整原生更新恢复验收。Stage2 另输出实际 MLP、Time、QKV、FlashAttention、SpConv、Pooling、Unpooling 的独立 AD 证据。
4. 对照身份审计。既有 B0 只在源/支持/训练数据/数据顺序/源 RNG/AMP/优化器/预算合同匹配，且 GPU 前向门槛通过后复用；EF1/EF2 不是 Flow-only 对照。
5. 独立32 successful-update Smoke，权重丢弃。正式重新加载源模型，100 → 500 → 10,000 updates **精确恢复同一优化器、EMA、scaler、source+interval RNG和总10K日程**。
6. 自动900 Development和 seq08/skip10041帧 NFE1/2/4/8配对评价。单个模型不按 NFE 重训。S1扫步数时 S2固定原始4；S2扫步数时 S1固定原始2。共享绝对 Parent ID 的高斯初态相同，FP32积分、终态才阈值。
7. 同卡同 AMP PyTorch 后端三轮交替 5+20 正式 Wall Benchmark；外部任务占卡则标记 `DEFERRED_EXCLUSIVE_GPU`，不把受干扰的质量运行时间当作加速结论。

任何失败保留负结果和堆栈，停止对应 Arm；另一个通过门槛的 Arm 独立继续。不会自动重试、重置训练或追加曝光。旧 EF 数值重放例外不继承；当前实现要求完整 native update replay 严格通过，未通过则停在工程门槛，不能宣称开训。

## 评价与筛选

使用原始 surface-hit GT、原900 Development、原冻结 seq0841帧 Manifest；seq08不调参。Native Fine Centers、全GT漏检 FN、Fine IoU/F1/P/R、Parent和 fixed-K ceiling、CD双向分量、修正物理 IoU0.5/0.2/0.1、JSD、观测/未知区域、距离/结构诊断、数量和最差帧全部保留。分序列/100帧时间块 bootstrap；只有一个训练 Seed，不宣称跨训练 Seed 稳定。

S1 MF1相对匹配 FM2 的 Parent IoU非劣界−0.3pp、Ceiling−0.5pp，下游几何不能明显退化。S2 MF1相对匹配 FM4 的 Fine IoU非劣界−0.5pp、CD相对退化≤2%，物理 IoU/JSD与覆盖不得明显恶化；最终仍需人工研究 GO 判断和真实时间证据。MF loss不能直接和 FM loss 数字排优劣，另记 diagonal velocity MSE。

现有 Shortcut 尚无完成且身份匹配的 Checkpoint，当前评价明确记为 `NOT_TESTED`，不伪造三方对照。待其独立完成后可按相同协议补充，不自动替换为 MeanFlow。

## 可复现入口

在本工作树、原 Python 环境：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -m pytest -q \
  tests/test_meanflow_math.py tests/test_meanflow_models.py tests/test_meanflow_runtime.py

# 只读检查，不启动 GPU 工作
CUDA_VISIBLE_DEVICES='' /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python \
  scripts/research_v2/hier_occ_flow/run_meanflow_queue.py \
  --queue configs/research_v2/hier_occ_flow/desescene_meanflow_queue.yaml \
  --output experiments/desescene_meanflow_v1/queue --once

# 实际控制器：先等待前序完成，再自动执行上述门槛与训练/评价
/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -u \
  scripts/research_v2/hier_occ_flow/run_meanflow_queue.py \
  --queue configs/research_v2/hier_occ_flow/desescene_meanflow_queue.yaml \
  --output experiments/desescene_meanflow_v1/queue
```

实际已启动的控制器见同目录 `supervisor.json`，不要重复启动。代码/config/hash变化会阻止后续任务启动，需要显式复核新身份。只提交代码、配置、紧凑审计/报告；不上传 Checkpoint、数据、全量输出或编译缓存。
