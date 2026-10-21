"""Inference-only TensorRT adapter for static dense-fidelity backbones."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class FidelityTensorRTBackbone(nn.Module):
    """Execute a static FP16 TensorRT backbone on the current PyTorch stream.

    The adapter intentionally does no host/device copies. Output buffers are
    reused after the prior backbone output has been consumed, which is the
    single-scene, sequential Stage1 encoder contract used by the pilot.
    """

    def __init__(self, engine_path: str | Path, *, device: torch.device | None = None):
        super().__init__()
        import tensorrt as trt

        self.engine_path = Path(engine_path).resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        self.engine_sha256 = hashlib.sha256(self.engine_path.read_bytes()).hexdigest()
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self._trt = trt
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {self.engine_path}")
        self._context = self._engine.create_execution_context()
        inputs, outputs = [], []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if inputs != ["features"] or outputs != [f"feature_{i}" for i in range(len(outputs))]:
            raise ValueError(f"Unexpected TensorRT IO contract: inputs={inputs}, outputs={outputs}")
        self.input_name, self.output_names = inputs[0], outputs
        if self._engine.get_tensor_dtype(self.input_name) != trt.float16:
            raise TypeError("Fidelity TensorRT engines must accept FP16 features")
        self.input_shape = tuple(self._engine.get_tensor_shape(self.input_name))
        if any(dim <= 0 for dim in self.input_shape):
            raise ValueError(f"Only static shapes are supported: {self.input_shape}")
        self.output_shapes, self.output_dtypes = [], []
        for name in outputs:
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = self._engine.get_tensor_dtype(name)
            if any(dim <= 0 for dim in shape) or dtype != trt.float16:
                raise TypeError(f"Expected static FP16 output, got {name}: {shape}, {dtype}")
            self.output_shapes.append(shape)
            self.output_dtypes.append(torch.float16)
        self._buffers = None

    def forward(self, features: torch.Tensor):
        if features.device != self.device or features.dtype != torch.float16:
            raise TypeError(f"Expected FP16 input on {self.device}, got {features.dtype} on {features.device}")
        if tuple(features.shape) != self.input_shape:
            raise ValueError(f"Input shape {tuple(features.shape)} != engine shape {self.input_shape}")
        if not features.is_contiguous():
            raise ValueError("Input must be contiguous; do explicit layout conversion at the model boundary")
        if self._buffers is None:
            self._buffers = tuple(torch.empty(shape, dtype=dtype, device=self.device)
                                  for shape, dtype in zip(self.output_shapes, self.output_dtypes))
        self._context.set_tensor_address(self.input_name, features.data_ptr())
        for name, output in zip(self.output_names, self._buffers):
            self._context.set_tensor_address(name, output.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution enqueue failed")
        return self._buffers
