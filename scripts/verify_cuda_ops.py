#!/usr/bin/env python3
"""Run a forward/backward smoke test for the bundled Chamfer CUDA op."""

import torch
from chamfer3D.dist_chamfer_3D import chamfer_3DDist


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with CUDA_VISIBLE_DEVICES=<target GPU>.")
    torch.manual_seed(0)
    first = torch.rand(2, 32, 3, device="cuda", requires_grad=True)
    second = torch.rand(2, 48, 3, device="cuda", requires_grad=True)
    dist1, dist2, _, _ = chamfer_3DDist()(first, second)
    loss = dist1.mean() + dist2.mean()
    loss.backward()
    if first.grad is None or second.grad is None:
        raise RuntimeError("Chamfer backward pass did not produce gradients.")
    print(f"[OK] Chamfer forward/backward passed on {torch.cuda.get_device_name(0)}; loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
