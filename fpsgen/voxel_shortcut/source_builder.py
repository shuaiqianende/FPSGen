"""Condition-only source rules; observation means direct input hits, not GT."""

import torch
from .count_mapping import normalize_density


def build_source(base_density, occupancy_probability, observed, valid, mapping, *, generator,
                 mode='observed_unknown_gaussian', sigma_observed=.01, sigma_unobserved=.05,
                 sigma_fixed=.05, sigma_floor=.01, support_threshold=.5, count_threshold=0.):
    if generator is None:
        raise ValueError("Explicit source generator required")
    if not 0 <= support_threshold <= 1 or count_threshold < 0:
        raise ValueError("Invalid frozen source gate")
    if occupancy_probability.shape != base_density.shape or observed.shape != base_density.shape:
        raise ValueError("Base probability/observation shape mismatch")
    if observed.dtype != torch.bool or not torch.isfinite(occupancy_probability).all() or bool(((occupancy_probability<0)|(occupancy_probability>1)).any()):
        raise ValueError("Input-hit bool mask and probability in[0,1] required")
    effective = base_density*((occupancy_probability>=support_threshold)&(base_density>=count_threshold))
    density = normalize_density(effective,valid)
    base = mapping.encode(density)
    epsilon = torch.randn(base.shape,device=base.device,dtype=torch.float32,generator=generator)
    if mode=='gaussian':
        state = epsilon
    elif mode=='deterministic':
        state = base
    elif mode=='fixed_gaussian':
        if sigma_fixed<=0:
            raise ValueError("Stochastic source requires positive noise")
        state = base+sigma_fixed*epsilon
    elif mode=='observed_unknown_gaussian':
        if min(sigma_observed,sigma_unobserved,sigma_floor)<=0:
            raise ValueError("Observation noise and floor must be positive")
        sigma = torch.where(observed,base.new_tensor(max(sigma_observed,sigma_floor)),base.new_tensor(max(sigma_unobserved,sigma_floor)))
        state = base+sigma*epsilon
    else:
        raise ValueError(f"Unknown source mode{mode}")
    # Invalid ROI is excluded from transport and remains zero-count mapped.
    state = torch.where(valid,state,state.new_tensor(-1))
    return state,{'base_density':density,'source_mode':mode,'gt_used':False,
                  'effective_support_voxels':int(((effective>0)&valid).sum())}
