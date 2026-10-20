# Compile and CUDA Graph Experiments

The benchmark entry point supports opt-in `stage2_mlp` and `stage1_flow`
compilation, records first-call wall time, and captures Dynamo graph counters.
PyTorch 2.0.1+cu117 and Triton are installed, but do not make every model path
compile-compatible.

## Stage2 MLP compile

The stage2 `local`, `time`, `fuse`, and `decoder` modules compiled and executed
on real frames. The first timed call (warmup frame 0) took 8.50 s due to lazy
compilation. Dynamo reported four unique graphs. On the 20-frame test,
generation averaged 750.18 ms and Stage2 averaged 321.86 ms, versus GPU3 eager
baseline 633.66 ms and 301.93 ms. This was slower and is rejected as a default
optimization. Dynamic token shapes and compile/runtime overhead outweigh any
small dense-MLP benefit in this configuration.

## Stage1 Flow compile

The one-frame smoke failed before completing inference. PyTorch 2.0.1 Inductor
raised `Inference tensors do not track version counter` while tracing
`frequencies[None]` in the Stage1 occupancy network's timestep embedding. No
fallback-to-eager was enabled, so the failure is explicit. This path is not
used in the recommended runtime.

CUDA Graph was not attempted. Stage2 token counts vary by scene and PTv3 uses
dynamic serialization, pooling and sparse-convolution metadata. Padding these
tokens without a proven masking scheme would change the model computation.

The next compile checks should run one real frame per target on an admitted
GPU, record lazy first-call compilation separately, then compare warmed 5+20
latency and graph/recompilation behavior. No compile speedup is claimed here.
