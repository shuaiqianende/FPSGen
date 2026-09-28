"""GPU1-only isolated environment gate: Torch, ME, Chamfer and compile."""
import json
from pathlib import Path
import torch

assert torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES=1"
result = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
          "cuda_available": torch.cuda.is_available(),
          "gpu": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported()}
x = torch.randn(64, 64, device="cuda", requires_grad=True)
(x @ x).mean().backward()
for dtype in (torch.float16, torch.bfloat16):
    with torch.autocast("cuda", dtype=dtype):
        y = (x @ x).mean()
    result[f"autocast_{dtype}"] = bool(torch.isfinite(y))
result["compile_dense"] = False
try:
    dense = torch.compile(torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.SiLU(), torch.nn.Linear(128, 64)).cuda())
    result["compile_dense"] = bool(torch.isfinite(dense(torch.randn(8, 64, device="cuda")).mean()))
except Exception as exc:
    result["compile_error"] = repr(exc)
try:
    import MinkowskiEngine as ME
    result["minkowski"] = getattr(ME, "__version__", "import OK")
except Exception as exc:
    result["minkowski_error"] = repr(exc)
try:
    from fpsgen.ops.chamfer import Chamfer3DDist
    p = torch.randn(1, 32, 3, device="cuda", requires_grad=True)
    d1, d2, _, _ = Chamfer3DDist()(p, p.detach().clone())
    (d1.mean() + d2.mean()).backward()
    result["chamfer_forward_backward"] = bool(torch.isfinite(p.grad).all())
except Exception as exc:
    result["chamfer_error"] = repr(exc)
Path("env/results").mkdir(parents=True, exist_ok=True)
Path("env/results/environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
