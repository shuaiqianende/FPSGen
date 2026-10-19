# CD 实验总览：GPU KD-tree 与 Chamfer CUDA for Scene

本页是提交给后续分析的阅读入口。此次仅整理 CD 算子、基准与小型证据；不修改训练、PointFlow、Sinkhorn、模型权重或数据集。

## 建议阅读顺序

1. [Scene 算法、接口、Teacher 实验与限制](CHAMFER_CUDA_FOR_SCENE_DESIGN.md)。
2. [最新统一规模对比](CHAMFER_CUDA_FOR_SCENE_SCALING.md)：11种点数 × 5种实现，共55组合、165次计时。
3. [早期三实现规模对比](CD_SCALING_1024_TO_1048576_GPU.md)：保留历史结果，不与后续轮次混拼。
4. [真实108万GT/18万预测点的时间与精度](CD_1080K_180K_GPU_TIMING_AND_PARITY.md)。
5. [18万点Teacher网络计时](TEACHER_GPU_KDTREE_CD_TIMING.md)与[108万GT context的Teacher计时](TEACHER_1080K_CONTEXT_GPU_TIMING.md)。最新五实现Teacher结果见第1项。
6. [早期GPU KD-tree逐点精度审计](EXACT_GPU_KDTREE_CD_AUDIT.md)与[显存、退化输入及失败门槛](GPU_KDTREE_TRADEOFFS_AND_MEMORY.md)。

## 统一定义与方法

CD = 0.5 × (预测到GT的最近邻欧氏距离均值 + GT到预测的最近邻欧氏距离均值)，单位m。
不是平方距离CD，也不是两方向均值直接相加。底层Scene接口返回平方距离，`sqrt_loss`负责上述归约。

| 实现 | 最近邻算法 | 精度与训练路径 |
|---|---|---|
| 原Chamfer CUDA | 全点对扫描，无需存储完整距离矩阵 | FP32，原始实现未修改 |
| Scene tiled | 独立动态CUDA网格、共享内存分块扫描 | FP32，原生坐标反向；仍为O(NM) |
| Scene BVH | Morton排序、平衡包围盒树、保守距离下界剪枝 | FP32，原生坐标反向 |
| KeOps | 现有在线距离归约实现 | 使用历史Teacher的CD约定 |
| 早期GPU KD-tree | 逐层XYZ分割排序、包围盒剪枝 | FP32坐标、FP64查询；PyTorch gather/norm反向 |

BVH和KD-tree都是成熟空间索引方法。这是场景规模的工程实现，不将使用BVH本身宣称为新算法。
不近似KNN、不限搜索半径、不限制访问节点数、不随机抽点、不构建密集N×M矩阵。
树的排序只组织索引，不替换原始XYZ；动态输入的建树时间包含在测试中。
变长场景分别搜索，不能把不同场景合并为同一最近邻集合。当前ragged接口逐场景调度，不代表已实现融合式多场景并行核。

## 最新结果摘要

GPU1 RTX3090，B1，预热后3次有效计时的中位数；具体warmup次数和流程以各脚本/原始记录为准。
规模测试是两份独立均匀三维体积点云，不是实际LiDAR表面。下表含CD前向和预测坐标反向，单位ms。

| 两侧各自点数 | 原CUDA | Scene tiled | Scene BVH | KeOps | GPU KD-tree |
|---:|---:|---:|---:|---:|---:|
|8,192|1.922|2.317|3.227|3.974|7.936|
|131,072|110.295|38.284|7.336|134.601|23.218|
|262,144|646.569|141.325|12.846|499.640|49.214|
|1,048,576|10318.401|2229.286|52.774|7898.867|136.295|

完整表、额外allocated显存、计时范围与原始轮次见[统一报告](CHAMFER_CUDA_FOR_SCENE_SCALING.md)。
百万对百万的195.52倍属于这次特定测试，不保证实际LiDAR或退化分布都达到。
Scene BVH百万点额外PyTorch allocated峰值约70.28MiB，原CUDA48MiB；不含公共输入或Teacher网络，非全进程nvidia-smi数值。

真实Teacher输入18万source +108万GT context，输出18万endpoint，网络FP16/CD FP32。
五种实现总计算中位数：原CUDA3.367s，tiled2.464s，BVH2.062s，KeOps3.482s，早期KD-tree2.122s。
不含优化器、DDP、I/O；网络反向约1.45s已成为主要开销。不能把CD核加速比例直接当作整网训练加速。

## 正确性：通过项与仍需保留的问题

- 最新55组标量CD相对CPU精确cKDTree参考最大绝对误差1.768e-7m；165次梯度均有限。
- 同一个最新Teacher endpoint上，各实现loss均为0.06746291369199753。Scene BVH相对KeOps梯度relative L2约8.704e-8，max absolute约1.819e-12，通过既定门槛。
- **历史失败不能抹去**：此前另一固定Teacher endpoint上，早期KD-tree梯度relative L2=4.539e-4，高于1e-4门槛。新一次通过不代表此前问题普遍解决。
- 最近邻等距点、FP32/FP64距离排序、原子累加均可能引入差异。标量接近不等于梯度逐位相等，也不证明长训收敛等价。
- 原CUDA在小点云更快；重叠包围盒、非零等距支持等分布会削弱剪枝。没有通用固定倍率保证。
- 新算子只承诺已测一阶坐标梯度，不承诺double backward；GPU原子累加不保证逐位确定性。
- 完整规模测试之后有一行极大坐标溢出边界修正，原始计时保留；最终50项测试和6组发布复核通过，文件hash保留在证据中。

## 代码与证据入口

- `fpsgen/ops/chamfer_scene.py`、`chamfer_scene_cuda.cu`：新Scene算子。
- `fpsgen/ops/gpu_kdtree.py`、`gpu_kdtree_cuda.cu`、`fpsgen/utils/gpu_kdtree_training.py`：早期GPU树与可微CD。
- `tests/test_chamfer_scene.py`、`test_gpu_kdtree.py`、`test_gpu_kdtree_training.py`：39项直接GPU算子测试；此次静态收集39项成功，历史真实GPU执行结果见审计JSON。
- `fpsgen/utils/gpu_kdtree_metrics.py`与`selected_point_metrics.py`是CD测试所需的最小评测依赖；保留其原有可选IoU能力，但此次不上传新的IoU/模型质量实验。
- `scripts/research_v2/hier_occ_inference/benchmark_chamfer_scene_scaling.py`：统一五实现测试。
- 同目录`benchmark_teacher_scene_cuda.py`：完整Teacher前向、CD和反向分段计时。
- 同目录`benchmark_cd_*`、`audit_cd_1080k_ties.py`、`benchmark_teacher_*`：历史规模、显存、tie和Teacher实验。
- [证据目录](stage2_morph_fm100k_audit/)：JSON、CSV、PNG；仅上传小型记录，不上传输入点云、checkpoint、环境或编译产物。
- [最新规模CSV](stage2_morph_fm100k_audit/chamfer_scene_scaling_summary.csv)、[曲线PNG](stage2_morph_fm100k_audit/chamfer_scene_scaling.png)、[发布审计](stage2_morph_fm100k_audit/chamfer_scene_delivery_audit.json)。

## 在原服务器复现

使用项目隔离环境；CUDA toolkit、缓存路径及GPU准入检查目前按本服务器配置。
Python3.9 / PyTorch2.0.1+cu117 / CUDA toolkit11.7 / RTX3090(sm86)，KeOps与原Chamfer扩展需可用。
Teacher和真实地图实验还需要原私有数据、历史Teacher checkpoint及已记录配置；这些大文件没有上传。
GPU忙碌时准入函数应拒绝运行，不能为了复现而停掉已有训练。

```bash
nvidia-smi --id=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.7
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export E2E_PYTHON=/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python
export KEOPS_CACHE_FOLDER=/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/teacher_cd_benchmark_keops_gpu1
"$E2E_PYTHON" -m pytest -q tests/test_chamfer_scene.py tests/test_gpu_kdtree.py tests/test_gpu_kdtree_training.py
"$E2E_PYTHON" scripts/research_v2/hier_occ_inference/benchmark_chamfer_scene_scaling.py --output /tmp/cd_scene_fresh_benchmark --repeats 3
"$E2E_PYTHON" scripts/research_v2/hier_occ_inference/benchmark_teacher_scene_cuda.py --output /tmp/cd_scene_fresh_teacher.json
```

选择不存在的output，脚本拒绝覆盖旧记录。新扩展编译缓存位于FPSGen/env内，不修改系统CUDA或原Chamfer源码。
复现入口与GPU准入仍引用仓库已有模块，不是脱离FPSGen的独立pip包。

## 与后续Sinkhorn分析的边界

此次没有修改或测试新的Sinkhorn方案。这里验证的是最近邻CD及其训练梯度。
Sinkhorn还包含候选边图、边权、行列归一化、稀疏归约和收敛诊断；CD的速度结果不能直接推导OT加速或等价性。
现有HS-OccFlow Stage2使用纯FM，不使用CD；此次没有自动切换任何正式训练后端。
