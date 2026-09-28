"""Inductor/Dynamo scope gate for dense code and a real ME sparse subgraph."""
import json
import re
from pathlib import Path

import torch
import MinkowskiEngine as ME


assert torch.cuda.device_count() == 1, "run with CUDA_VISIBLE_DEVICES=1"
result = {"dense_compile": False, "me_compile_forward": False}

dense = torch.nn.Sequential(
    torch.nn.Linear(128, 256), torch.nn.LayerNorm(256), torch.nn.SiLU(), torch.nn.Linear(256, 128)
).cuda()
x_dense = torch.randn(64, 128, device="cuda")
try:
    compiled_dense = torch.compile(dense, backend="inductor")
    result["dense_compile"] = bool(torch.isfinite(compiled_dense(x_dense)).all())
except Exception as exc:
    result["dense_error"] = repr(exc)

# TensorField-style coordinates are created in FP32 and remain outside compile.
coords = ME.utils.batched_coordinates(
    [torch.randint(0, 32, (4096, 3), dtype=torch.int32)], dtype=torch.float32
).cuda()
x_sparse = ME.SparseTensor(features=torch.randn(4096, 4, device="cuda"), coordinates=coords)
sparse = torch.nn.Sequential(
    ME.MinkowskiConvolution(4, 16, kernel_size=3, dimension=3),
    ME.MinkowskiBatchNorm(16),
    ME.MinkowskiReLU(),
).cuda()
try:
    compiled_sparse = torch.compile(sparse, backend="inductor")
    sparse_out = compiled_sparse(x_sparse).F
    torch.cuda.synchronize()
    result["me_compile_forward"] = bool(torch.isfinite(sparse_out).all())
except Exception as exc:
    result["me_compile_error"] = repr(exc)

# A successful callable can still be all graph breaks. Record Dynamo's graph
# evidence separately rather than claiming sparse Inductor acceleration.
try:
    # PyTorch 2.0 uses the legacy two-argument explain form; newer versions
    # support explain(model)(inputs). Keep this probe pinned to 2.0.1.
    explained = torch._dynamo.explain(sparse, x_sparse)
    if isinstance(explained, tuple):  # PyTorch 2.0 legacy explain return.
        summary, _guards, graphs, ops_per_graph, break_reasons, _verbose = explained
        result["me_dynamo_summary"] = summary
        result["me_graph_count"] = len(graphs)
        match = re.search(r"(\d+) graph break", summary)
        result["me_graph_break_count"] = int(match.group(1)) if match else None
        result["me_captured_op_count"] = sum(len(ops) for ops in ops_per_graph)
        result["me_break_reasons"] = [str(reason) for reason in break_reasons[:5]]
    else:
        result["me_graph_count"] = int(explained.graph_count)
        result["me_graph_break_count"] = int(explained.graph_break_count)
        result["me_break_reasons"] = [str(reason) for reason in explained.break_reasons[:5]]
        result["me_captured_op_count"] = None
    # The wrapper can execute after torch.compile while ME custom ops stay
    # eager. Only captured operations can constitute an Inductor speed target.
    result["me_inductor_effective"] = bool(result["me_captured_op_count"])
except Exception as exc:
    result["me_explain_error"] = repr(exc)

Path("env/results").mkdir(parents=True, exist_ok=True)
Path("env/results/compile_probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
