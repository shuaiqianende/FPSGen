"""FP16/BF16 ME probe; coordinates remain FP32, only network compute is AMP."""
import argparse
import torch
import MinkowskiEngine as ME

p = argparse.ArgumentParser(); p.add_argument("--dtype", choices=("fp16", "bf16"), required=True); a = p.parse_args()
dtype = torch.float16 if a.dtype == "fp16" else torch.bfloat16
if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported")
coords = ME.utils.batched_coordinates([torch.randint(0, 32, (4096, 3), dtype=torch.int32)], dtype=torch.float32).cuda()
features = torch.randn(4096, 4, device="cuda")
model = torch.nn.Sequential(ME.MinkowskiConvolution(4, 16, 3, dimension=3), ME.MinkowskiBatchNorm(16), ME.MinkowskiReLU()).cuda()
opt = torch.optim.Adam(model.parameters(), 1e-3); scaler = torch.cuda.amp.GradScaler(enabled=dtype is torch.float16)
for step in range(100):
    x = ME.SparseTensor(features=features, coordinates=coords)
    with torch.autocast("cuda", dtype=dtype): out = model(x).F.square().mean()
    opt.zero_grad(); scaler.scale(out).backward(); scaler.step(opt); scaler.update(); assert torch.isfinite(out), step
print(f"{a.dtype} ME PASS", float(out), "coord_dtype", coords.dtype)
