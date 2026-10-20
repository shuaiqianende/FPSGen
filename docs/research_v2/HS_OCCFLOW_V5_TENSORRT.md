# HS-OccFlow V5 TensorRT Status

TensorRT execution is **NOT TESTED / BLOCKED by the available environment**. The active PyTorch environment is `2.0.1+cu117`; TensorRT, Torch-TensorRT, ONNX, ONNX Runtime, and `trtexec` were absent from the inspected environment paths. No Engine was built, no checkpoint or network was modified, and no speed or parity claim is made.

The measured convolution profile shows dense 3D convolutions are a reasonable future target, but the existing cuDNN path already selected an Ampere FP16 Tensor-Core-capable convolution kernel for a representative large layer. A useful TRT comparison must therefore run on the same GPU/runtime and include GPU-resident handoff costs. Installing or switching frameworks was outside this completed validation, so there is no TRT number in the V5 results.
