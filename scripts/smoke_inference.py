#!/usr/bin/env python3
"""Run one real preprocessed FPSGen sample through BEV and PointFlow inference."""

import argparse
import os

import numpy as np

from fpsgen.inference import DiffCompletion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input_/frame.npy")
    parser.add_argument("--student", required=True, help="stage-3 checkpoint")
    parser.add_argument("--bev", required=True, help="stage-1 checkpoint")
    parser.add_argument("--image-steps", type=int, default=1)
    parser.add_argument("--point-steps", type=int, default=1)
    args = parser.parse_args()

    os.environ["FPSGEN_BEV_CHECKPOINT"] = args.bev
    scan = np.load(args.input)[:, :3]
    model = DiffCompletion(args.student, "", args.point_steps, 2.0)
    completed, _ = model.complete_scan(
        (scan, None, args.input), steps=[args.image_steps, args.point_steps]
    )
    if completed.ndim != 2 or completed.shape[1] != 3 or len(completed) == 0:
        raise RuntimeError(f"Invalid generated point cloud shape: {completed.shape}")
    print(f"[OK] End-to-end inference generated {len(completed)} XYZ points.")


if __name__ == "__main__":
    main()
