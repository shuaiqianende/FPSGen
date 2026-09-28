"""100-iteration FP32 MinkowskiEngine forward/backward stability gate."""
import torch
import MinkowskiEngine as ME

assert torch.cuda.device_count() == 1
coords = ME.utils.batched_coordinates([torch.randint(0, 32, (4096, 3), dtype=torch.int32)], dtype=torch.float32).cuda()
features = torch.randn(4096, 4, device="cuda")
model = torch.nn.Sequential(ME.MinkowskiConvolution(4, 16, 3, dimension=3), ME.MinkowskiBatchNorm(16), ME.MinkowskiReLU()).cuda()
opt = torch.optim.Adam(model.parameters(), 1e-3)
for step in range(100):
    # Coordinates are intentionally FP32 and outside any autocast context.
    x = ME.SparseTensor(features=features, coordinates=coords)
    out = model(x).F.square().mean(); opt.zero_grad(); out.backward(); opt.step()
    assert torch.isfinite(out), step
print("FP32 ME PASS", float(out))
