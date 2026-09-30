# FPSGen Research V2 — 项目索引

更新日期：2026-10-01。本文是 Research V2 的**导航与状态索引**；它不替代每项实验的原始配置、日志或结果文档。

## 工作区规则

- 源码、配置、清单、小型 CSV/JSON 汇总和 Markdown 文档可以进入 Git。
- `experiments/`、`outputs/`、`env/conda/`、编译缓存、点云、checkpoint 和 TensorBoard 日志均为机器本地产物；不得提交、搬迁或删除，除非单独确认。
- 原始数据目录中的 `gt_/`、`input_/`、`velodyne/`、`map_clean.npy`、`poses.txt`、`calib.txt` 是只读基线。新采样 GT 仅位于各 sequence 的 `gt_possion/`。
- `gt_possion` 是既定目录名，保留该拼写以保持训练和数据路径兼容。
- 当前有 DDP Student 训练正在使用 `experiments/` 与 `outputs/research_v2/student_train/`；整理期间不移动这些路径。

## 当前正式运行

| 项目 | 当前约定 |
| --- | --- |
| Stage-3 Student | DCD Teacher target + sparse Sinkhorn refinement |
| Student config | `configs/research_v2/train_student_dcd_sinkhorn_gt_possion_10ep_ddp2_bs2.yaml` |
| 训练数据 | SemanticKITTI 00–07、09、10；`gt_dir: gt_possion` |
| DDP | 物理 GPU 2、3；每 rank batch 2；全局 batch 4 |
| Sinkhorn target | `K=8`，`epsilon=0.002`，`iterations=100`，`alpha=1.0` |
| Teacher checkpoint | `experiments/teacher_dcd_gt_possion_ft1ep_ddp2_bs2_lr1e4/lightning_logs/version_0/checkpoints/teacher_dcd_gt_possion_ft1ep_ddp2_bs2_lr1e4_epoch=00.ckpt` |
| Student log | `outputs/research_v2/student_train/logs/student_dcd_sinkhorn_gt_possion_10ep_ddp2_bs2.log` |
| Student checkpoints | `experiments/student_dcd_sinkhorn_gt_possion_10ep_ddp2_bs2/lightning_logs/version_0/checkpoints/` |
| Launcher | `scripts/launch_student_dcd_sinkhorn_formal.sh` |

`scripts/launch_student_dcd_sinkhorn_formal.sh` 与上述 10-epoch config 是本机尚未入 Git 的活跃文件；在训练结束前不得重命名、移动或覆盖。

## 研究主线与结论入口

| 方向 | 状态 | 入口 |
| --- | --- | --- |
| Baseline / 进度记录 | 进行中 | [progress/README.md](progress/README.md) |
| Teacher–Student source-row feature alignment | Gate A 已通过 | [FEATURE_GUIDANCE_ANALYSIS.md](FEATURE_GUIDANCE_ANALYSIS.md) |
| Teacher-guided Sparse Sinkhorn endpoint refinement | 已完成参数诊断；当前 Student 使用选定配置 | [SINKHORN_ENDPOINT_ANALYSIS.md](SINKHORN_ENDPOINT_ANALYSIS.md)、[SINKHORN_ITERATION_ABLATION.md](SINKHORN_ITERATION_ABLATION.md) |
| DCD-only Teacher | 已训练并作为当前候选 Teacher | [DCD_TEACHER_5EPOCH_ANALYSIS.md](DCD_TEACHER_5EPOCH_ANALYSIS.md) |
| LiDiff-compatible fixed-voxel + Hard-Poisson GT | 已生成/QA；当前训练数据契约 | [GT_POISSON_GENERATION.md](GT_POISSON_GENERATION.md)、[GT_POSSSION_TRAINING_CONTRACT.md](GT_POSSSION_TRAINING_CONTRACT.md) |
| Stage-1 LiDAR-only BEV 评价 | 代码与协议已准备 | [BEVFLOW_LIDAR_ONLY_EVALUATION.md](BEVFLOW_LIDAR_ONLY_EVALUATION.md) |
| Stage-1 DiC-S / PixelU-S backbone | 已有训练动态结果；DiC-S 候选、PixelU-S HOLD | [BEV_BACKBONE_DIC_PIXELU.md](BEV_BACKBONE_DIC_PIXELU.md)、[BEV_BACKBONE_TRAINING_DYNAMICS.md](BEV_BACKBONE_TRAINING_DYNAMICS.md)、[backbone_sources.yaml](backbone_sources.yaml) |
| Stage-1 HDiT-S / DiP-S / NCSNpp-S backbone | C0 五轮基线：HDiT/DiP 已完成、NCSNpp 运行中；C1–C4 条件注入：HDiT/DiP C1 运行中 | [BEV_BACKBONE_HDIT_DIP_NCSNPP.md](BEV_BACKBONE_HDIT_DIP_NCSNPP.md)、[BEV_CONDITION_INJECTION_STUDY.md](BEV_CONDITION_INJECTION_STUDY.md) |
| Teacher speed environment / AMP / Inductor | 独立工程实验 | [TEACHER_SPEED_ENV_PROBE.md](TEACHER_SPEED_ENV_PROBE.md) |

## 可复现配置与清单

### 数据和评测清单

- `configs/research_v2/gate_seq08_10.txt`
- `configs/research_v2/gate_seq08_20.txt`
- `configs/research_v2/gate_seq08_100.txt`
- `configs/research_v2/eval_bevflow_lidar_only.yaml`

这些 manifest 是 sequence 08 固定帧集合；涉及 Teacher、Sinkhorn、BEV 或 Student 的横向比较时不得重新随机抽帧。

### Teacher

- `configs/research_v2/train_teacher_dcd_a1_l1_5ep*.yaml`：DCD-only Teacher 的不同启动形式。
- `configs/research_v2/finetune_teacher_dcd_gt_possion_1ep_ddp2_bs2.yaml`：切换到 `gt_possion` 的一 epoch 微调。

### Student

- `configs/research_v2/train_student_dcd_sinkhorn_gt_possion_500_ddp2_bs2.yaml`：短程稳定性配置。
- `configs/research_v2/train_student_dcd_sinkhorn_gt_possion_10ep_ddp2_bs2.yaml`：当前正式训练配置。
- `configs/research_v2/feature_f*.yaml`：Feature Guidance 的独立消融；不得与当前 Sinkhorn Student 混用。

### Stage-1 BEV

- `configs/research_v2/train_bev_legacy_gt_possion.yaml`
- `configs/research_v2/train_bev_dic_s_gt_possion.yaml`
- `configs/research_v2/train_bev_pixelu_s_gt_possion.yaml`
- `configs/research_v2/train_bev_hdit_s_gt_possion.yaml`
- `configs/research_v2/train_bev_dip_s_gt_possion.yaml`
- `configs/research_v2/train_bev_ncsnpp_s_gt_possion.yaml`
- `configs/research_v2/condition_ablation/phase1.yaml`：Stage-1 条件注入
  C1–C4 的统一清单；具体运行配置由协调器从 C0 配置物化，避免手工复制。

三份配置都使用 `gt_possion`；它们的唯一区别应是 BEV backbone，不应借此混入不同的数据或训练目标。

## 代码入口

### 核心模型与算子

- `fpsgen/models/gen_teacher.py`：Teacher；支持 DCD-only objective。
- `fpsgen/models/gen_student.py`：Student PointFlow；支持 target refinement 与可选 feature guidance。
- `fpsgen/ops/dcd.py`：训练安全的 DCD 实现。
- `fpsgen/ops/endpoint_refinement/`：KNN、Sparse Sinkhorn 与近似 OT target refinement。
- `fpsgen/models/bev_backbones/`：legacy / DiC-S / PixelU-S / HDiT-S / DiP-S / NCSNpp-S 的 lazy factory 和 dense BEV backbone。

### 数据、诊断和评测

- `scripts/generate_semantickitti_poisson_gt.py`：生成 `gt_possion`。
- `scripts/audit_semantickitti_voxel020.py`：固定体素 candidate 数审计。
- `scripts/diagnose_feature_correspondence.py`、`scripts/run_feature_gate.py`：Feature alignment Gate。
- `scripts/cache_teacher_endpoints.py`、`scripts/run_sinkhorn_gate.py`、`scripts/run_direct_source_sinkhorn.py`：Teacher / Direct Sinkhorn 诊断。
- `scripts/eval_teacher_endpoint.py`、`scripts/eval_teacher_distribution.py`：几何与局部点分布评测。
- `scripts/eval_bevflow_lidar_only.py`：Stage-1 仅 LiDAR 的三通道 BEV 评测。
- `scripts/eval_bev_condition_usage.py`：固定 sequence 08 的条件正确/置零/循环错配使用率评测。
- `scripts/run_bev_condition_campaign.py`、`scripts/summarize_bev_condition_campaign.py`：C1–C4 训练协调与可审计汇总。
- `scripts/infer_student_oracle_bev.py`：不运行 BEVFlow 的 Student-only oracle-BEV 推理。
- `scripts/render_student_trajectory_bev_video.py`：Student trajectory 的三视图视频渲染。
- `scripts/inspect_bev_backbone.py`：CPU 参数量与 backbone 合约检查。

## 本地产物地图（不进入 Git）

| 目录 | 内容 | 保留策略 |
| --- | --- | --- |
| `experiments/` | Lightning checkpoint、events、训练状态 | 正在使用；仅训练完成后按实验人工归档 |
| `outputs/research_v2/dcd_teacher/` | DCD Teacher endpoint / distribution 结果 | 保留原始汇总与可视化 |
| `outputs/research_v2/sinkhorn_diagnostic/` | Sinkhorn 参数消融结果 | 保留；文档只引用汇总 |
| `outputs/research_v2/sinkhorn_cache/` | P0、Teacher endpoint、GT 的离线 cache | 不提交、不移动 |
| `outputs/research_v2/gt_poisson_generation/` | Poisson 生成与 QA 汇总 | 保留；原始 GT 位于数据根目录 |
| `outputs/research_v2/feature_correspondence/` | Gate A 对齐统计 | 保留 |
| `outputs/research_v2/feature_train/` | Feature Guidance 短训练产物 | 保留，独立于当前 Student |
| `outputs/research_v2/direct_source_sinkhorn/` | P0 直接 Sinkhorn 诊断与 PLY | 保留为诊断基线 |
| `outputs/research_v2/bev_backbone_speed/` | BEV backbone 速度记录 | 保留小型结果；原始 Inductor cache 不提交 |
| `outputs/condition_campaign/` | 条件注入 smoke、配置物化、同步吞吐、使用率与汇总 | 活跃；不提交、不移动 |
| `outputs/research_v2/student_train/` | 当前 Student 文本日志 | 活跃，禁止清理 |
| `outputs/research_v2/student_oracle_bev/` | Student-only PLY、轨迹与视频 | 保留推荐结果与历史可视化变体 |
| `env/inductor_cache/`、`env/keops_cache_legacy/` | 环境特定编译 cache | 不提交；仅在确认不再需要后清理 |

## 推荐定位入口

### 当前 Student 的最新可检查产物

- Checkpoint 目录：`experiments/student_dcd_sinkhorn_gt_possion_10ep_ddp2_bs2/lightning_logs/version_0/checkpoints/`
- 最新已导出的 Student-only 单帧结果：`outputs/research_v2/student_oracle_bev/latest_epoch02_002713_cfg1_s50/`
  - `source_oracle_bev.ply`
  - `student_cond100.ply`
  - `gt_possion.ply`
  - `lidar_scan.ply`
- 推荐的密集 LiDiff-style trajectory 视频：`outputs/research_v2/student_oracle_bev/trajectory_video_002713_cfg1_s50/trajectory_bev_lidiff_dense.mp4`

注意：该 trajectory 视频源自此前保存的 trajectory states；不同 epoch 的 final inference 必须单独以其目录和 manifest 识别，不能将视频误标为最新 checkpoint 的结果。

## 整理后的使用约定

1. 新的正式实验必须先新增 `configs/research_v2/<experiment>.yaml`，并在对应分析 Markdown 记录 commit、seed、数据契约和 checkpoint。
2. 机器本地结果统一写入 `outputs/research_v2/<experiment>/`；checkpoint 继续由 Lightning 放入 `experiments/<experiment>/`。
3. 原始大文件、PLY、cache 和环境目录只在本机保留；Git 只提交可审阅的代码、配置、文档与小型汇总。
4. 训练进行中不重命名其配置、log、checkpoint 或输出目录。训练结束后再决定是否把历史可视化变体压缩或归档。
5. 添加新成果时优先更新对应专题文档和 `progress/`；本索引只维护入口与状态。
