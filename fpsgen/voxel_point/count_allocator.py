"""PWM-style stochastic residual allocation, without its uniform fallback."""

import torch


class InvalidDensityError(ValueError):
    pass


def allocate_counts(density, spec, *, generator, total_points=180000, minimum_mass=1e-12):
    if generator is None:
        raise ValueError("A recorded torch.Generator is required")
    if total_points < 1:
        raise ValueError("Positive point count required")
    if tuple(density.shape) != spec.shape:
        raise InvalidDensityError("Density must have XYZ grid shape")
    if not bool(torch.isfinite(density).all()):
        raise InvalidDensityError("NaN/Inf density")
    weights = spec.dense_to_flat(density.double().clamp_min(0) * spec.valid_circle_mask(device=density.device))
    mass = weights.sum()
    if float(mass) <= minimum_mass:
        raise InvalidDensityError("Valid density mass is zero/near-zero")
    expected = weights / mass * total_points
    counts = expected.floor().long()
    remaining = total_points - int(counts.sum())
    if remaining < 0:
        raise InvalidDensityError("Allocation roundoff produced excess mass")
    if remaining:
        residual = expected - counts
        extra = torch.multinomial(residual, remaining, replacement=True, generator=generator)
        counts.scatter_add_(0, extra, torch.ones_like(extra))
    if int(counts.sum()) != total_points or bool((counts < 0).any()):
        raise InvalidDensityError("Integer allocation invariant failed")
    return counts
