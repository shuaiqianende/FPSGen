#!/usr/bin/env python3
"""Cache frozen-Teacher endpoint tuples once for Sinkhorn-only sweeps."""
import argparse
from pathlib import Path
import numpy as np
import torch
from run_sinkhorn_gate import FrozenTeacher, _manifest

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--sequence', default='08')
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
        gt = torch.from_numpy(np.load(a.dataset_root / a.sequence / 'gt_' / f'{stem}.npy')[:, :3]).float().cuda()
        p0, endpoint = teacher.endpoint(gt, a.resolution)
        torch.save({'p0': p0.cpu(), 'endpoint': endpoint.cpu(), 'gt': gt.cpu(),
                    'seed': a.seed + i, 'frame': stem}, a.output / f'{stem}.pt')
        print(f'cached {stem}', flush=True)

if __name__ == '__main__':
    main()
