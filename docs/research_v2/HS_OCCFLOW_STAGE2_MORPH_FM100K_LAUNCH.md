# Stage2 新方案100K样本训练启动记录

本报告启动准备阶段，正式运行状态以实际日志、`controller_status.json`和`launch_verification.json`为准。

## 固定实验

- 分支：`research/hs-occflow-384-stage2-score-gtparents`；正式代码版本：`d686d15`。
- 用户授权：“可以用这个数据集，然后我跟你说的那个新方案开始训练吧”。
- 预算：100,000次成功样本曝光；globalbatch4（GPU2/3、每卡2个完整变长场景），从零共25,000次成功更新。工程Smoke不计入正式预算。
- Stage2：2,677,128参数、64D Sparse PTv3、每父体素一个token、8维局部Flow；纯有效子体素FM MSE，Gaussian sigma1，target=2occ-1。无BCE/Shortcut/Chamfer等训练项。
- Stage1：保留固定最终100K-frame EMA checkpoint，SHA`b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`；实际384三层特征为128×16×192×192、128×8×96×96、128×4×48×48。每场景Stage1编码一次，4 NFE生成实际score；score不是校准概率。
- Parent score输入实际Stage1终点得分，取消常数1。训练全部GT非空父体素均保留，score不用于删掉GT支持。
- 额外空父体素比例0；内部未占用子格仍监督为0。与真实预测父体素集合的分布差异仍需单独测试，不能将Oracle模型质量当作部署质量。
- 新目标：26邻域一次膨胀→腐蚀，在原GT父体素范围内并入原占用。原点命中不删除，粗父体素不增加；新增格子是增强标签。
- Stage2从随机初始化开始，不加载旧Stage2、pilot或smoke的模型/优化器。完整joint Stage1不再训练，Stage1参数冻结。
- AdamW lr2e-4、weight_decay.01、clip1、FP16 AMP、activation checkpointing、FlashAttention、1000步warmup+cosine、EMA.999。
- 自动development评测按用户之前要求关闭。原始地图评价入口仍保留，seq08 skip100仅在用户明确请求时测评，不用于选参。

## 数据与安全

原始train17,870、development900、guard360协议不变，训练不含seq08。完整新数据保存在`experiments/hier_occ_flow/occ384_morph_close26_training_v1/`。

预处理仅调整CPU worker上限由8到32，先使用24worker、随后按同一处理器身份断点续做32worker；闭运算算法、footprint、几何均不变。新processor SHA在YAML中显式绑定。原4worker生成器通过SIGINT退出，其已完成shard及日志留存于旧新标签目录，没有停止任何训练任务。

## 启动与工程门槛

```bash
tmux attach -t fpsgen_hs_stage2_morph_fm_gb4_gpu23
```

控制器依次执行：完整train/development生成与校验→GPU2/3各自实际新标签B2最大场景反向、打包/排列/Flash/SpConv/重计算梯度检查→双卡连续21与20+恢复1对照→继续至32成功更新→新随机Stage2正式启动。

不复用旧gate。所有GPU证据绑定代码、geometry、Stage1权重、数据manifest、closing transform和processor SHA。任何工程失败停止此流程并留存日志；没有用旧模型回退。完整新数据未完成时只显示等待，不计作正式训练。

预检目录：`experiments/hier_occ_flow/stage2_real_score_gt_only_morph/preflight/`。
正式输出：`experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/`。

正式step0保存online、EMA、optimizer、scaler、scheduler、各rank RNG及sampler；step20和100另存早期checkpoint，之后每500步latest、每2500步里程碑。启动审计必须验证online与初始EMA相同、优化器初始为空、local/PTv3/score/decoder参数真实改变、两rank同步、实际80次样本曝光和有限非零梯度。
