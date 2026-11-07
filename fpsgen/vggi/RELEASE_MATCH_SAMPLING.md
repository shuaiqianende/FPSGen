# VGGI：当前软匹配与连续概率场采样入口

本发布包含经过验证的软匹配、连续平方振幅采样器及所需依赖，训练模型和数据缓存留在服务器。
详细结果、限制和复现命令见 [优化报告](docs/OPTIMIZED_MATCH_AND_SAMPLING_20261108.md)。

```python
from fpsgen.vggi.point_transport import match
from fpsgen.vggi.field_generation.sample_cuda import CUDAFieldSampler

sampler = CUDAFieldSampler(amplitude, origin, spacing)
source, sample_stats = sampler.sample(180000, seed=20261107)
endpoint, match_stats = match(source, gt_xyz)
velocity_target = endpoint - source
```

`match` 默认支持 FP32 CUDA `[N,3]` 或等长 `[B,N,3]`，保持各场景独立。
使用精确动态KNN、纯软Sinkhorn梯度和连续端点，绝无硬匹配回退。
默认K32、3次坐标更新、每次30轮、ε=(.02,.005,.002)、λ=50、步长.9。
有限迭代的局部解尚不保证全局最优，也不代表训练质量已验收。

`amplitude` 为XYZ顺序的带符号振幅系数，连续密度为B-spline插值之后的平方。
若网络输出Gram状态q，先复用几何合同对应的 `AmplitudeCodec.decode_cuda(q)`。
GPU采样精确返回180K个点，不加噪声，不做阈值过滤；计算精度FP32。

运行环境：已在Python3.9、PyTorch2.0.1+CUDA11.7、RTX3090验证。
额外需要numpy、scipy、ninja、pytest，以及与PyTorch兼容的NVCC/C++工具链。
首次调用编译CUDA算子；热启动测速不包含首次编译。

```bash
CUDA_VISIBLE_DEVICES=3 python -m pytest -q \
  tests/test_vggi_batch_soft.py tests/test_vggi_linear_graph.py \
  tests/test_vggi_bvh_neighbors.py tests/test_vggi_soft_graph.py \
  tests/test_vggi_dynamic_soft.py tests/test_vggi_cuda_sampling.py \
  tests/test_vggi_amplitude_sampling.py
```

真实数据基准脚本按本地 `fpsgen/vggi/runs/` 缓存目录寻找三帧数据；发布不包含KITTI数据或模型权重。
独立单元测试使用合成小场景，不依赖KITTI。旧算子以显式 `reference_match` 供数值对照。
