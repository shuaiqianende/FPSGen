"""Persistent, identity-checked cache of the exact pre-Poisson raw GT crop.

This only avoids regenerating the same immutable crop for each evaluated
model. It does not subsample, voxelize, change order, or approximate points.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .raw_gt import RawMapSource
from .storage import atomic_json, file_sha

ROOT = Path(__file__).resolve().parents[2]


class ExactRawGTCache:
    def __init__(self, data_root, geometry, output, *, sequence='08'):
        self.data_root = Path(data_root).resolve()
        self.geometry = geometry
        self.sequence = str(sequence)
        sequence_dir = self.data_root / self.sequence
        self.code_paths = [Path(__file__), ROOT / 'fpsgen/hier_occ_flow/raw_gt.py',
            ROOT / 'fpsgen/hier_occ_flow/viewpoint_lookup.py', ROOT / 'fpsgen/hier_occ_flow/geometry.py',
            ROOT / 'scripts/prepare_semantickitti.py', ROOT / 'scripts/generate_semantickitti_poisson_gt.py']
        identity = {'sequence': self.sequence, 'data_root': str(self.data_root),
            'geometry_sha256': geometry.sha256, 'fast_viewpoint': True,
            'sources': {name: file_sha(sequence_dir / name) for name in ('map_clean.npy', 'poses.txt', 'calib.txt')},
            'code': {str(p.relative_to(ROOT)): file_sha(p) for p in self.code_paths}}
        self.identity = identity
        self.sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.root = Path(output) / self.sha256
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = None
        manifest = self.root / 'identity.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Raw GT cache identity mismatch')
        atomic_json(manifest, identity)

    def load(self, frame):
        frame = int(frame)
        scan = self.data_root / self.sequence / 'velodyne' / f'{frame:06d}.bin'
        scan_sha = file_sha(scan)
        points_path = self.root / f'{frame:06d}.npy'
        meta_path = self.root / f'{frame:06d}.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['scan_sha256'] != scan_sha or meta['identity_sha256'] != self.sha256:
                raise ValueError('Raw GT source changed after caching')
            if not points_path.exists() or file_sha(points_path) != meta['points_file_sha256']:
                raise ValueError('Raw GT cached points missing or corrupted')
            target = np.load(points_path, allow_pickle=False)
            if target.dtype != np.float32 or list(target.shape) != meta['shape']:
                raise ValueError('Raw GT cached array shape/dtype changed')
            return target
        if self.source is None:
            self.source = RawMapSource(self.data_root, fast_viewpoint=True)
        raw = self.source.load(self.sequence, frame)
        _, valid = self.geometry.voxelize_points(raw)
        target = np.ascontiguousarray(raw[valid], dtype=np.float32)
        if target.ndim != 2 or target.shape[1] != 3 or not len(target) or not np.isfinite(target).all():
            raise ValueError('Empty or invalid raw GT crop')
        temp = self.root / f'.{frame:06d}.{os.getpid()}.tmp'
        with temp.open('wb') as f:
            np.save(f, target, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, points_path)
        atomic_json(meta_path, {'identity_sha256': self.sha256, 'scan_sha256': scan_sha,
            'points_file_sha256': file_sha(points_path), 'shape': list(target.shape), 'dtype': 'float32'})
        return target
