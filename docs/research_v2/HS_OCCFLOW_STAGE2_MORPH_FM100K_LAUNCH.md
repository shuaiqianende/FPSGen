# Stage2 新方案100K样本训练启动记录

本轮正式训练已经实际启动，启动后参数更新和双卡同步审计通过。以下快照仅为启动报告，100K样本预算尚未完成。

## 固定实验

- 分支：`research/hs-occflow-384-stage2-score-gtparents`；正式代码版本：`1d29848`。
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


## 实际执行证据：RUNNING

- 完整新数据：train17,870、development900，生成与校验均COMPLETE。新增细占用分别37.7944%/35.6478%，父体素ID和所有原占用保留。train平均5.3411/8子格占用，全8格占用父体素比例37.48%；数据是形态学增强标签，非额外传感器测量。
- 新train manifest SHA：`b0f79f5d5ec1a2ff94bdb80edf79af7ea7f92600319af8ea8659e0b41b7d7270`；development：`7e2b85a7c9d9cdeb672678eea08ca37e6b1a3e7d8d60dc85818057cd6dd94023`。
- CPU回归91PASS、1项CUDA检查在CPU命令中SKIP；随后GPU2/3实际新标签B2最大场景、Flash/SpConv、排列/打包/重计算梯度检查全部PASS。每卡最大639,476tokens，峰值allocated12.840GiB/reserved14.199GiB。
- 双卡32次成功更新已实际执行，AMP跳步0；连续21对照20+恢复1的模型/EMA/优化器/scaler/RNG/sampler检查PASS，未放宽原预注册容差。
- 短smoke FM loss：约1.9996→1.8924，仅工程学习迹象，不是几何质量验证。smoke权重未接入正式训练。
- 正式模型从随机参数重新启动：step0 optimizer为空、EMA=online；step20有163个可训练参数张量改变，local/trunk/score/decoder等模块均发生实际更新。
- 正式快照：65次成功更新，260/100,000次成功样本曝光，FM=1.983377，AMP跳步0。
- 近期平均2.945s/update，据启动期小样本外推剩余约20.4小时；尚未稳定覆盖全场景分布，不能当作精确完工时间。
- torchrun PID：328886，rank0 PID：329325；物理GPU2/3，未终止其他GPU作业。
- 实际初始化model/EMA SHA均为`4b1f9a7d5da30692371fdb855370599287621f2ad7caddac2651cc76076205b9`。
- 初始checkpoint SHA：`93909a1ecd9a66b209a18abd077ec1a2d4241025e1636b4e46e524f1d4365705`；step20 SHA：`1a8c18d57230eabb46380c18efb2a5629422b1ae4fe2198588890fb78cb073da`。
- 配置SHA：`baf889fc3f864f8f7100fb0f4874bbad70304798242647facf0fe8191bfa6681`；source SHA：`85bdbb777dc08c827f24b39208027bbc28dce813c1fc863896fc995498830d48`；launch Git：`1d298483f23badf7263e1b71a6d3a72231a53a85`。
- 已保存step0/20及resume_latest完整训练状态。固定1000步warmup+cosine，500步保存latest，2500步保存里程碑，100步另有早期checkpoint。
- 小型完整证据见 `stage2_morph_fm100k_audit/`。不提交数据、PLY、环境和模型权重。

### 本次工程故障及修复

提高CPU并行度的断点续做首次因JSON几何列表/内存元组比较失败，未覆盖已完成7822帧。已规范化JSON比较并新增真实CLI fresh+resume回归测试，之后成功续做至完整数据。旧4worker部分生成目录及日志保留，正式训练使用新的`occ384_morph_close26_training_v1`，未修改旧训练结果。
