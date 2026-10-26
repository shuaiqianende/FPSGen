# Encoder 20K / 900帧结果与 MeanFlow GPU 门槛

快照：2026-10-26 13:20:13 +0800。本报告不把独立工程预检称为正式训练。

## 已完成的900帧评价

EF1、EF2、Flow-only均完成10K successful updates / 20K exposures，以及900帧原协议完整评价。源任务及评价未被停止或覆盖。当前冻结参考EF0仍在原GPU2进程运行，已完成 **216/900**。

| 方案 | Parent IoU，NFE2 | 部署Fine IoU，S1=2/S2=4 | CD m↓ |
|---|---:|---:|---:|
| Flow-only | 46.2399% | 31.0324% | 0.209893 |
| EF1 | 46.2768% | 31.0379% | 0.208172 |
| EF2 | 46.2406% | 31.0347% | 0.206825 |

对照为同样追加20K的Flow-only，不是尚未完成的EF0。结果是新Encoder仅带来毫米级平均CD改善，Fine IoU没有清楚提高，当前不能宣布找到明显更好的Encoder，也不能据此进入100K。

分序列/100帧时间块的配对bootstrap（单训练Seed）中，EF1和EF2的部署Fine IoU差值95% CI均跨0。完整统计见 `meanflow/evidence_20261026/encoder_20k_900_interim.json`。最终EF0 GO判定及新的独占性能测量尚未完成。

评价边界：既有部署评价器按候选行抽取Stage2高斯源，候选集合变化时公共Parent噪声未严格按绝对ID对齐；因此微小Fine变化不能仅归因Encoder。Teacher-Support对照的900帧Parent-ID/input hash全部匹配，适合隔离特征变化。MeanFlow后续评价器已注册绝对Parent-ID噪声对齐。

## GPU3独立工程预检

EF1/EF2和Flow-only评价释放卡后，先在空闲GPU3执行工程检查；这没有提前启动正式20K，也没有影响GPU2的EF0。两个预检使用原固定源Hash、原精度、真实最大Batch2，没有截断Parent。

**S2-MF20K：BLOCKED_JVP。**

- 两个真实场景Parent数：192,179和191,976，总384,155。
- 三个不同interval的源前向零初始化对照逐位一致；缓存前向、打乱Parent顺序也逐位一致。
- local MLP、time MLP、QKV Linear、unpool投影/gather的native JVP通过。
- SpConv embedding/CPE在functorch包装下通过`data_ptr()`访问无storage Tensor失败。
- FlashAttention的autograd.Function缺`setup_context`；Pooling/segment-max也在同一native AD路径失败。
- 完整动态PTv3的native JVP未通过，因此没有进入Smoke或正式训练。没有使用假零、有限差分、非Flash替换或删除算子。

**S1-MF20K：BLOCKED_ENGINEERING，定位到 native_end_to_end_resume_replay。**

- 真实完整Batch2的Conv3d/GroupNorm/插值native JVP有限且非零；完整正常反向和更新已执行。
- 原输出零初始化、cached forward逐位一致；Checkpoint/optimizer/EMA/scaler及RNG的磁盘恢复逐位一致。
- 恢复后重新执行原生更新未逐位一致：两次loss相同，gradient norm为7.7191791534与7.7191829681。
- 这还不能直接称为合格Numerical Replay：需要自身同后端未修改FM基线及完整Tensor误差界审计，不能继承EF的数值例外。

紧凑原始结果/完整算子堆栈保存在 `meanflow/evidence_20261026/S1-MF20K_gpu_preflight.json`、`S2-MF20K_gpu_preflight.json` 和 component probe 文件。未通过完整门槛的方案不标GPU_PASS。

## 后续状态

原MeanFlow控制器PID930632仍等待EF0结束，源代码SHA保持1293e8bef20b2684859fac32db4b228eaed0b604a93b8143fc6274c1ab617391；没有热改排队源代码。上述预检的文件与正式队列路径分离，原日志和Checkpoint均保留。

MeanFlow正式训练：**NOT STARTED**；MeanFlow质量/速度：**NOT TESTED**。后续优先审计Stage1原生恢复差异；Stage2需要兼容原算子数学语义的JVP适配，不能绕过失败直接跑20K。即使EF0完成，也必须先通过对应工程门槛。Stage2监督Encoder保持USER_HOLD，未启动新Seed、100K或额外架构训练。
