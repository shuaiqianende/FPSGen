#!/usr/bin/env python
"""Build a static FP16 TensorRT engine for a fidelity-pilot backbone export."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu", type=int, choices=(2, 3), default=2)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--workspace-gib", type=int, default=4)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly the selected physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(a.onnx.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_gib << 30)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("Selected GPU reports no fast FP16 support")
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    build_start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - build_start
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    a.engine.parent.mkdir(parents=True, exist_ok=True)
    a.engine.write_bytes(serialized)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Could not deserialize just-built engine")
    context = engine.create_execution_context()
    input_name = "features"
    if engine.get_tensor_dtype(input_name) != trt.float16:
        raise RuntimeError(f"Unexpected engine input dtype {engine.get_tensor_dtype(input_name)}")
    host_input = np.load(a.reference_dir / "real_input.npy", mmap_mode="r")
    if tuple(host_input.shape) != tuple(engine.get_tensor_shape(input_name)):
        raise ValueError("Saved real input shape differs from engine input shape")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        input_tensor = torch.from_numpy(np.array(host_input, copy=True)).to("cuda", non_blocking=True)
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    output_tensors = {}
    for name in output_names:
        dtype = engine.get_tensor_dtype(name)
        if dtype != trt.float16:
            raise RuntimeError(f"Expected FP16 output from backbone, {name} has {dtype}")
        shape = tuple(engine.get_tensor_shape(name))
        with torch.cuda.stream(stream):
            output_tensors[name] = torch.empty(shape, device="cuda", dtype=torch.float16)
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    for name, tensor in output_tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())
    for _ in range(a.warmups):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT warm-up enqueue failed")
    stream.synchronize()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_ms, cuda_ms = [], []
    for _ in range(a.repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream.synchronize()
        start = time.perf_counter()
        start_event.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        end_event.record(stream)
        stream.synchronize()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - start) * 1000)
        cuda_ms.append(float(start_event.elapsed_time(end_event)))

    errors = []
    for index, name in enumerate(output_names):
        expected = np.load(a.reference_dir / f"pytorch_output_{index}.npy", mmap_mode="r")
        actual = output_tensors[name].cpu().numpy()
        if actual.shape != expected.shape:
            raise ValueError(f"Output {name} shape {actual.shape} != reference {expected.shape}")
        delta = actual.astype(np.float32) - np.asarray(expected, dtype=np.float32)
        denom = max(float(np.linalg.norm(np.asarray(expected, dtype=np.float32).ravel())), 1e-12)
        errors.append({"name": name, "shape": list(actual.shape),
                       "max_abs": float(np.max(np.abs(delta))),
                       "mean_abs": float(np.mean(np.abs(delta))),
                       "relative_l2": float(np.linalg.norm(delta.ravel()) / denom),
                       "finite": bool(np.isfinite(actual).all())})
    inspector = engine.create_engine_inspector()
    try:
        inspector_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    except Exception as exc:
        inspector_json = None
        inspector_error = repr(exc)
    else:
        inspector_error = None
    gpu = torch.cuda.get_device_properties(0)
    report = {
        "status": "ENGINE_BUILT_AND_RUN",
        "physical_gpu": a.physical_gpu,
        "gpu": gpu.name,
        "compute_capability": [gpu.major, gpu.minor],
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "precision_request": "fp16",
        "onnx": str(a.onnx.resolve()),
        "onnx_bytes": a.onnx.stat().st_size,
        "engine": str(a.engine.resolve()),
        "engine_bytes": a.engine.stat().st_size,
        "build_seconds": build_seconds,
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "output_names": output_names,
        "cuda_ms": {"mean": float(np.mean(cuda_ms)), "median": float(np.median(cuda_ms)),
                    "p90": float(np.percentile(cuda_ms, 90)), "p95": float(np.percentile(cuda_ms, 95)),
                    "values": cuda_ms},
        "wall_ms": {"mean": float(np.mean(wall_ms)), "median": float(np.median(wall_ms)),
                    "p90": float(np.percentile(wall_ms, 90)), "p95": float(np.percentile(wall_ms, 95)),
                    "values": wall_ms},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "output_errors_vs_pytorch_fp16": errors,
        "engine_inspector_json": inspector_json,
        "engine_inspector_error": inspector_error,
        "notes": [
            "Input/output tensors are GPU resident; reported engine wall time includes host enqueue and CUDA completion, excludes loading and CPU/GPU input copies.",
        "This is an FP16-only engine build; it does not perform PTQ or claim INT8 execution.",
        "The engine inspector report records the selected implementation precision per layer.",
        ],
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "tensorrt_version", "precision_request",
        "build_seconds", "engine_bytes", "cuda_ms", "wall_ms", "peak_allocated_gib",
        "output_errors_vs_pytorch_fp16")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
