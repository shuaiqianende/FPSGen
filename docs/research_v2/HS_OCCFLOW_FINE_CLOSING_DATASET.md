# Fine Occupancy 形态学闭运算数据版本（2026-10-18）

## 已实现并实测

在768×768×64 fine网格执行一次膨胀→腐蚀，比较6邻域十字结构元素和26邻域3×3×3结构元素。选择26邻域作为单独的新标签实验版本 `hs-occ-morph-close-v1`，不是经过验证的质量最优方法。

公式：`augmented = original OR (closing(original) AND valid_child AND original_parent_domain)`。

原始占用全部保留，仅在既有GT非空父体素内部增加子格，父体素ID、粗占用和索引不变。不是把8格全填满。XY分块64个父体素、每边2个fine格halo；不保存全量dense fine grid，跨块结果与全网格SciPy closing一致。球形ROI外不新增格子。默认SciPy1.10.1。

26邻域新增格子在这8帧到原占用格中心的最大距离为0.226131m；真实地图之外没有额外测量证据，新格子均为形态学增强标签。闭运算可能连接原本独立的邻近表面，尚需可视化与后续模型训练检验。

| Split | Frame | 6-neighbor added | 26-neighbor added |
|---|---|---:|---:|
| train | 02/004289 | 16.46% | 34.58% |
| train | 06/000444 | 10.18% | 29.59% |
| train | 02/003360 | 21.26% | 41.39% |
| train | 02/000925 | 25.23% | 47.78% |
| development | 05/000554 | 12.05% | 30.56% |
| development | 06/000555 | 11.04% | 32.72% |
| development | 02/000518 | 23.19% | 44.65% |
| development | 02/000515 | 23.39% | 44.26% |

## 可视化样本

训练seq02/004289：137,999个父体素不变；细占用617,932→831,599（新增213,667，34.58%）。

- 原始PLY：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_real_score_gt_only/closing_preview_seq02_004289/original_fine.ply`
- 补密PLY：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_real_score_gt_only/closing_preview_seq02_004289/closed_fine.ply`
- 仅新增格PLY：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_real_score_gt_only/closing_preview_seq02_004289/added_only.ply`
- 局部对照图片：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_real_score_gt_only/closing_preview_seq02_004289/closing_comparison.png`

这些是GT标签诊断，不是新模型推理效果。

## 全量数据任务：RUNNING，不是已生成完成

- 独立tmux：`fpsgen_hs_occ384_closing26_data`；4个CPU worker，不使用GPU。
- 目标：train17,870 + development900；不读取seq08。
- 新目录：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/occ384_morph_close26_parent_preserving_v1`。
- 日志：`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_real_score_gt_only/prepare_closing26.log`。
- 报告快照时train已原子完成416帧/13shards；development尚未开始。实时状态读各split的`build_state.json`，不能据此快照宣称全量完成。
- 按当前速度估计全量需约2–4小时，负载变化会影响ETA。
- 新uint8 child masks理论额外2,752,659,008 bytes（约2.75GB/2.56GiB），parent IDs和index硬链接复用不变的数据，避免重复存储约11GB。源文件只读，标签不会覆盖原始数据。
- shard原子提交；中断后显式`--resume`验证源manifest/geometry/transform/processor SHA与已完成shard哈希。全部shards完成且校验后才发布可加载的最终manifest。
- processor SHA：`522d31fb4e0543e9b398f28334103f22485a951979580fc2ac92e18829a85395`；transform SHA：`1c88afa56324775ecc3a56f7720d0f2d69b38b7ee9ac7f925b67b9dcfaae3802`。

## 训练与评价合同

新morph YAML显式绑定数据版本、源manifest及处理器SHA，旧配置默认只接受raw v1，禁止静默混用。继续使用实际Stage1 score、GT parents only、FM-only。

新版本的Stage2验证主指标仍对原始raw-map fine GT计算，额外记录`augmented_target_fine`，避免仅因修改GT使数字看起来更好。未启动新正式长训练；需完成新数据一致性、DDP恢复和学习效果检查。

## 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python \
scripts/research_v2/hier_occ_flow/prepare_fine_closing.py \
 --config configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_smoke.yaml \
 --output experiments/hier_occ_flow/occ384_morph_close26_parent_preserving_v1 \
 --workers 4 --connectivity 26
# 断点继续时添加 --resume；不要并行启动同一输出的第二个生成器。
```
