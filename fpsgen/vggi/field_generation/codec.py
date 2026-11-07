"""FP32 reference codec for squared-amplitude fields, NOT positive B3 weights.

Finite-domain raw bases psi have no per-node normalization. G=R.T R;
z=R a/sqrt(scale), density=(sum a*psi)^2, mass=scale*||z||^2.
"""
from itertools import product
import numpy as np
from scipy.linalg import solve_banded

FORMAT = 'vggi_squared_amplitude_gram_v1'


def b3(t):
    t = np.abs(t)
    return np.where(t < 1, 2/3-t*t+t*t*t/2,
                    np.where(t < 2, (2-t)**3/6, 0))


class AmplitudeCodec:
    def __init__(self, shape, origin, spacing, scale):
        self.shape = tuple(shape)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.spacing = np.asarray(spacing, dtype=np.float32)
        self.scale = float(scale)
        if (len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
                or self.origin.shape != (3,) or self.spacing.shape != (3,)
                or not np.isfinite(self.origin).all() or not np.isfinite(self.spacing).all()
                or not (self.spacing > 0).all() or not np.isfinite(scale) or scale <= 0):
            raise ValueError('invalid fixed field geometry/scale')
        self.factors = []
        # Four-point Gauss integrates each degree-six product exactly, up to
        # FP32 arithmetic. Split at all cardinal spline knots and domain edges.
        x, w = np.polynomial.legendre.leggauss(4)
        x, w = x.astype(np.float32), w.astype(np.float32)
        for n, h in zip(shape, self.spacing):
            gram = np.zeros((n, n), dtype=np.float32)
            for j in range(n):
                for k in range(j, min(n, j+4)):
                    edges = sorted(set([0., float(n)] +
                        [float(np.clip(v+.5+q, 0, n)) for v in (j, k) for q in range(-2, 3)]))
                    value = np.float32(0)
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        p = (lo+hi)/2 + (hi-lo)/2*x
                        value += np.float32((hi-lo)/2)*np.dot(w, b3(p-j-.5)*b3(p-k-.5))
                    gram[j, k] = gram[k, j] = h*value
            self.factors.append(np.linalg.cholesky(gram).T.astype(np.float32))

    def _field(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != self.shape or not np.isfinite(value).all():
            raise ValueError('expected finite field array in XYZ order')
        return value

    def encode(self, amplitude):
        out = self._field(amplitude)
        for axis, r in enumerate(self.factors):
            a = np.moveaxis(out, axis, 0)
            result = np.zeros_like(a)
            for offset in range(min(4, len(r))):
                length = len(r)-offset
                result[:length] += r.diagonal(offset).reshape(-1, 1, 1)*a[offset:]
            out = np.moveaxis(result, 0, axis)
        return out/np.float32(np.sqrt(self.scale))

    def decode(self, state):
        out = self._field(state)*np.float32(np.sqrt(self.scale))
        for axis, r in enumerate(self.factors):
            n = len(r)
            band = np.zeros((4, n), dtype=np.float32)
            for offset in range(min(4, n)):
                band[3-offset, offset:] = r.diagonal(offset)
            a = np.moveaxis(out, axis, 0)
            back = solve_banded((0, 3), band, a.reshape(n, -1)).reshape(a.shape)
            out = np.moveaxis(back, 0, axis)
        return out

    def decode_cuda(self, state):
        """Decode one XYZ Gram state on its GPU; reuse this codec across frames."""
        import torch
        from .sample_cuda import extension
        if not isinstance(state, torch.Tensor) or not state.is_cuda or tuple(state.shape)!=self.shape:
            raise ValueError('Expected CUDA tensor with codec XYZ shape')
        with torch.no_grad():
            q=state.float().contiguous()
            if not bool(torch.isfinite(q).all()):raise ValueError('Nonfinite Gram state')
            factors=[torch.as_tensor(r,device=q.device).contiguous() for r in self.factors]
            a=extension().decode(q,factors,self.scale)
            if not bool(torch.isfinite(a).all()):raise FloatingPointError('Nonfinite decoded amplitude')
            return a

    def mass(self, state):
        z = self._field(state)
        # Extended precision for a scalar diagnostic reduction, not Teacher KDE.
        return self.scale*float(np.sum(z.astype(np.float64)**2))

    def density(self, amplitude, xyz):
        a = self._field(amplitude)
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('expected finite query XYZ[N,3]')
        u = (xyz-self.origin)/self.spacing-.5
        lower = np.floor(u).astype(np.int64)-1
        f = np.zeros(len(xyz), dtype=np.float32)
        inside = ((xyz >= self.origin) &
                  (xyz < self.origin+self.spacing*np.asarray(self.shape))).all(1)
        for offset in product(range(4), repeat=3):
            indices = lower+np.asarray(offset)
            valid = inside & ((indices >= 0) & (indices < np.asarray(self.shape))).all(1)
            ids = indices[valid]
            f[valid] += a[ids[:, 0], ids[:, 1], ids[:, 2]]*b3(u[valid]-ids).prod(1)
        return f*f
