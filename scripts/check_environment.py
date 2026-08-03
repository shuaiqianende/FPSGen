#!/usr/bin/env python3
"""Fail fast when the FPSGen CUDA environment is incomplete."""

import importlib
import sys


REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "MinkowskiEngine",
    "pytorch_lightning",
    "open3d",
    "pykeops",
    "chamfer_3D",
)


def main() -> None:
    failures = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "local")
            print(f"[OK] {name}: {version}")
        except Exception as error:
            failures.append(f"{name}: {error}")

    try:
        import torch
        print(f"[INFO] PyTorch CUDA build: {torch.version.cuda}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        print(f"[INFO] visible GPU count: {torch.cuda.device_count()}")
        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable to PyTorch")
        elif torch.cuda.device_count() != 1:
            print("[WARN] Expected one GPU after CUDA_VISIBLE_DEVICES remapping.")
        else:
            print(f"[OK] selected GPU: {torch.cuda.get_device_name(0)}")
    except Exception as error:
        failures.append(f"PyTorch CUDA probe: {error}")

    if failures:
        print("\n[FAIL] Environment is not ready:", *failures, sep="\n  - ", file=sys.stderr)
        raise SystemExit(1)
    print("\n[OK] FPSGen environment check passed.")


if __name__ == "__main__":
    main()
