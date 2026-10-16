"""Occasional GPU-side gradient aggregates and small parameter-change samples."""
import hashlib
import torch


def module_grad_norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def gradient_stats(module):
    params=list(module.parameters());grads=[p.grad.detach().float() for p in params if p.grad is not None]
    if not grads: return {'l2':0.,'abs_max':0.,'nonfinite_elements':0,'parameters_with_gradient':0,'nonzero_parameter_ratio':0.}
    l2=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
    maximum=torch.stack([g.abs().max() for g in grads]).max()
    invalid=torch.stack([(~torch.isfinite(g)).sum() for g in grads]).sum()
    nonzero=torch.stack([(g!=0).any().float() for g in grads]).sum()
    return {'l2':float(l2) if bool(torch.isfinite(l2)) else None,'abs_max':float(maximum) if bool(torch.isfinite(maximum)) else None,
        'nonfinite_elements':int(invalid),'parameters_with_gradient':len(grads),'nonzero_parameter_ratio':float(nonzero)/max(len(params),1)}


def sample_parameters(module):
    return torch.cat([p.detach().float().flatten()[:64] for p in module.parameters()]).clone()


def sample_changes(module,previous):
    sample=sample_parameters(module); difference=sample-previous
    return sample,{'sample_delta_l2':float(difference.norm()),'sample_changed_elements':int((difference!=0).sum()),
        'sample_elements':sample.numel(),'sample_sha256':hashlib.sha256(sample.cpu().numpy().tobytes()).hexdigest()}
