"""Diagnostic candidate: process-local PyTorch2.0 baddbmm correction.

ATen beta=0 must ignore the additive input, even if it contains NaN/Inf.
The installed 2.0 decomposition instead evaluates input*0 + bmm. No torch
installation files, model weights, objective or precision policy are changed.
This fixes the isolated operation probe, NOT the observed full BEV CFG1
failure. Do not treat this module as the production recovery for that failure.
"""
import numbers

import torch


def safe_baddbmm(additive, left, right, beta=1, alpha=1):
    result = torch.bmm(left, right)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if isinstance(beta, numbers.Number) and beta == 0:
        return result
    if not isinstance(beta, numbers.Number) or beta != 1:
        additive = additive * beta
    return additive + result


def install():
    """Correct only the affected2.0 decomposition, before compilation."""
    if not torch.__version__.split('+')[0].startswith('2.0.'):
        raise RuntimeError('This compatibility fix is restricted to audited PyTorch2.0')
    from torch._inductor import decomposition
    operation = torch.ops.aten.baddbmm.default
    original = decomposition.decompositions[operation]
    if original is safe_baddbmm:
        return
    decomposition.decompositions[operation] = safe_baddbmm
    # select_decomp_table caches a merged copy, including on prior compiles.
    decomposition.fast_random_decomps.cache_clear()


def main():
    import json
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('Numerical probe uses physicalGPU3 only')
    generator = torch.Generator(device='cuda').manual_seed(42)
    left = torch.randn(2,8,16,device='cuda',dtype=torch.float16,generator=generator)
    right = torch.randn(2,16,8,device='cuda',dtype=torch.float16,generator=generator)
    additive = torch.full((2,8,8),float('nan'),device='cuda',dtype=torch.float16)
    def operation(a,l,r):
        return torch.baddbmm(a,l,r,beta=0)
    eager = operation(additive,left,right)
    compiled = torch.compile(operation,backend='inductor')(additive,left,right)
    install()
    torch._dynamo.reset()
    fixed = torch.compile(operation,backend='inductor')(additive,left,right)
    print(json.dumps(dict(torch_version=torch.__version__,
                          eager_finite=bool(torch.isfinite(eager).all()),
                          original_inductor_finite=bool(torch.isfinite(compiled).all()),
                          fixed_inductor_finite=bool(torch.isfinite(fixed).all()),
                          fixed_max_abs_error=float((fixed-eager).abs().max())),indent=2),flush=True)


if __name__ == '__main__':
    main()
