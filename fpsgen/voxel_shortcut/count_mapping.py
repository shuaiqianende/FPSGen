"""FP32 symmetric-log mapping without clipping targets at n_ref."""

import math
import torch


class InvalidDensityError(ValueError):
    pass


class CountMapping:
    def __init__(self, n_ref, *, max_decode_count=1000000.):
        if not math.isfinite(n_ref) or n_ref <= 0 or max_decode_count <= n_ref:
            raise ValueError("n_ref>0 and larger finite decode safety bound required")
        if not math.isfinite(max_decode_count):
            raise ValueError("Finite decode bound required")
        self.n_ref, self.max_decode_count = float(n_ref),float(max_decode_count)
        self.log_ref = math.log1p(n_ref)

    def encode(self, counts):
        n = counts.float()
        if not torch.isfinite(n).all() or bool((n<0).any()):
            raise InvalidDensityError("Invalid count mapping input")
        return 2*torch.log1p(n)/self.log_ref-1

    def decode(self, state, *, return_diagnostics=False):
        x = state.float()
        if not torch.isfinite(x).all():
            raise InvalidDensityError("Nonfinite mapped field")
        exponent = (x+1)*(.5*self.log_ref)
        cap = math.log1p(self.max_decode_count)
        counts = torch.expm1(exponent.clamp(max=cap)).clamp_min(0)
        if return_diagnostics:
            return counts,{'decode_exponent_capped_fraction':float((exponent>cap).float().mean()),
                           'negative_count_clipped_fraction':float((x < -1).float().mean())}
        return counts


def normalize_density(counts, valid, *, point_budget=180000, min_mass=1e-8):
    p = counts.float()
    if p.ndim != 5 or p.shape[1]!=1 or not torch.isfinite(p).all() or bool((p<0).any()):
        raise InvalidDensityError("Finite nonnegative[B,1,Z,Y,X] density required")
    p = p*valid
    mass = p.sum((1,2,3,4),keepdim=True)
    if bool((mass<=min_mass).any()):
        raise InvalidDensityError("Zero effective density; uniform fallback prohibited")
    return p*(point_budget/mass)
