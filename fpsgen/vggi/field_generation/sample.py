"""Direct rejection sampling of rho=(sum a_j B3_j)^2 on the finite domain.

Signed coefficients are not mixture probabilities. A cell proposal uses a
conservative bound: |sum a_j B3_j| <= max(active |a_j|), since the nonnegative
cardinal bases sum to at most one after finite-domain truncation. A base cell
intersects supports of at most five nodes per axis (four at any given point).
No extra Gaussian, threshold, or voxel-center density approximation is used.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from .codec import AmplitudeCodec, FORMAT, b3


def density(amplitude, xyz, origin, spacing):
    """Vectorized 64-node evaluation; accumulate in FP64 for rejection tests."""
    a = np.asarray(amplitude)
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    shape = np.asarray(a.shape)
    u = (xyz-origin)/spacing-.5
    lower = np.floor(u).astype(np.int64)-1
    ids = [lower[:, d, None]+np.arange(4) for d in range(3)]
    basis = [b3(u[:, d, None]-ids[d]) *
             ((ids[d] >= 0) & (ids[d] < shape[d])) for d in range(3)]
    ids = [np.clip(ids[d], 0, shape[d]-1) for d in range(3)]
    value = np.zeros(len(xyz), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            xy = basis[0][:, i]*basis[1][:, j]
            for k in range(4):
                value += a[ids[0][:, i], ids[1][:, j], ids[2][:, k]]*xy*basis[2][:, k]
    inside = ((xyz >= origin) & (xyz < origin+shape*spacing)).all(1)
    return value*value*inside


class FieldSampler:
    def __init__(self, amplitude, origin, spacing):
        self.a = np.asarray(amplitude, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.spacing = np.asarray(spacing, dtype=np.float64)
        if (self.a.ndim != 3 or min(self.a.shape) < 1 or
                not np.isfinite(self.a).all() or self.origin.shape != (3,) or
                self.spacing.shape != (3,) or not np.isfinite(self.origin).all() or
                not np.isfinite(self.spacing).all() or not (self.spacing > 0).all()):
            raise ValueError('Expected finite amplitude and positive XYZ geometry')
        bound = maximum_filter(np.abs(self.a), size=5, mode='constant', cval=0)
        self.upper = bound.astype(np.float64).ravel()**2*(1+1e-12)
        self.cdf = np.cumsum(self.upper)
        self.total = float(self.cdf[-1])
        if not np.isfinite(self.total) or self.total <= 0:
            raise ValueError('Cannot sample an empty/nonfinite field')

    def sample(self, count=180000, *, seed=20261107, batch_size=65536,
               max_proposals=100000000, progress=None):
        if not isinstance(count, int) or count < 1 or batch_size < 1:
            raise ValueError('Positive sample count and batch size required')
        rng = np.random.default_rng(seed)
        out = np.empty((count, 3), dtype=np.float64)
        filled = proposed = accepted = 0
        start = time.monotonic()
        while filled < count:
            n = min(batch_size, max_proposals-proposed)
            if n <= 0:
                raise RuntimeError('Rejection proposal budget exhausted; no partial output returned')
            flat = np.searchsorted(self.cdf, rng.random(n)*self.total, side='right')
            cell = np.stack(np.unravel_index(flat, self.a.shape), axis=1)
            xyz = self.origin+(cell+rng.random((n, 3)))*self.spacing
            rho = density(self.a, xyz, self.origin, self.spacing)
            upper = self.upper[flat]
            if not np.isfinite(rho).all() or (rho > upper*(1+1e-10)).any():
                raise FloatingPointError('Density exceeds proposal bound')
            keep = rng.random(n)*upper < rho
            chosen = xyz[keep]
            take = min(len(chosen), count-filled)
            out[filled:filled+take] = chosen[:take]
            filled += take; proposed += n; accepted += len(chosen)
            if progress:
                progress(dict(points=filled, target=count, proposals=proposed,
                              seconds=time.monotonic()-start))
        return out, dict(points=count, proposals=proposed, acceptance=accepted/proposed,
                         envelope_mass=self.total*float(np.prod(self.spacing)),
                         seconds=time.monotonic()-start, seed=seed,
                         method='continuous_squared_amplitude_cell_envelope_rejection_v1',
                         extra_noise=False, threshold=None)


def load_cache(path, *, device='cpu'):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f['metadata']))
        if meta['format'] != FORMAT:
            raise ValueError('Expected squared amplitude Gram cache')
        codec = AmplitudeCodec(meta['shape'], meta['origin'], meta['spacing'], 1.)
        q = f['q']
        a = codec.decode(q)
        mass = codec.mass(q)
    if str(device)=='cpu':sampler=FieldSampler(a,meta['origin'],meta['spacing'])
    else:
        from .sample_cuda import CUDAFieldSampler
        sampler=CUDAFieldSampler(a,meta['origin'],meta['spacing'],device=device)
    return sampler,meta,mass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--points', type=int, default=180000)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--device',default='cuda',help='cuda / cuda:0 / cpu; CUDA is the recommended fast sampler')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sampler, meta, mass = load_cache(args.cache,device=args.device)
    xyz, info = sampler.sample(args.points, seed=args.seed)
    if hasattr(xyz,'cpu'):xyz=xyz.cpu().numpy()
    info.update(scene_id=meta['scene_id'], cache=str(args.cache.resolve()),
                field_mass=mass, expected_acceptance=mass/info['envelope_mass'],
                contract_sha256=meta['contract_sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.save(f, xyz)
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
