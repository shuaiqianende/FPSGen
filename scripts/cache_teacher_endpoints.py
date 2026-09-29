#!/usr/bin/env python3
"""Cache frozen-Teacher endpoint tuples once for Sinkhorn-only sweeps."""
import argparse
from pathlib import Path
import numpy as np
import torch
from run_sinkhorn_gate import FrozenTeacher, _manifest
from fpsgen.datasets.dataloader.semantic_kitti import load_preprocessed_points

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--sequence', default='08')
    p.add_argument(
        '--gt-dir', default='gt_',
        help='Full-scene target directory; accepts historical gt_ NPY or gt_possion PLY.',
    )
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--teacher-ckpt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=float, default=.05)
    p.add_argument('--seed', type=int, default=20260928)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Teacher endpoint cache requires CUDA')
    a.output.mkdir(parents=True, exist_ok=True)
    teacher = FrozenTeacher(a.teacher_ckpt, torch.device('cuda'))
    for i, stem in enumerate(_manifest(a.manifest)):
        torch.manual_seed(a.seed + i)
        gt_path_npy = a.dataset_root / a.sequence / a.gt_dir / f'{stem}.npy'
        gt_path_ply = a.dataset_root / a.sequence / a.gt_dir / f'{stem}.ply'
        if gt_path_npy.is_file():
            gt_path = gt_path_npy
        elif gt_path_ply.is_file():
            gt_path = gt_path_ply
        else:
            raise FileNotFoundError(
                f'No target for frame {stem} in {a.dataset_root / a.sequence / a.gt_dir}'
            )
        gt = torch.from_numpy(load_preprocessed_points(str(gt_path))[:, :3]).float().cuda()
        p0, endpoint = teacher.endpoint(gt, a.resolution)
        torch.save({'p0': p0.cpu(), 'endpoint': endpoint.cpu(), 'gt': gt.cpu(),
                    'seed': a.seed + i, 'frame': stem, 'gt_dir': a.gt_dir,
                    'gt_path': str(gt_path)}, a.output / f'{stem}.pt')
        print(f'cached {stem}', flush=True)

if __name__ == '__main__':
    main()
