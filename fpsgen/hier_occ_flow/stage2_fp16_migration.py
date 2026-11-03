"""Explicit opt-in FP32-checkpoint to FP16-runtime state migration.

EMA stays in FP32 storage; callers must update it from masters, not half weights.
This helper does not enable compilation or change geometry/conditioning.
"""
import copy
import torch
from .explicit_fp16_training_policy import FP32Masters, apply_explicit_policy


def migrate_fp32_runtime(model,optimizer,ema_state):
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Migrate before casting; expected FP32 model')
    before={n:v.detach().clone() for n,v in model.state_dict().items()}
    ema={n:v.detach().clone() for n,v in ema_state.items()}
    if ema.keys()!=before.keys():raise ValueError('EMA model state differs')
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in ema.values()):
        raise ValueError('EMA must be FP32 before migration')
    masters=FP32Masters(model,before)
    optimizer_state=copy.deepcopy(optimizer.state_dict())
    groups=masters.optimizer_groups(optimizer.param_groups)
    new_optimizer=type(optimizer)(groups,**optimizer.defaults)
    new_optimizer.load_state_dict(optimizer_state)
    policy=apply_explicit_policy(model)
    for name,p in model.named_parameters():
        if not torch.equal(p,masters.params[name].to(p.dtype)):raise ValueError('Master rounding mismatch')
    for state in new_optimizer.state.values():
        if any(torch.is_tensor(v) and v.is_floating_point() and v.dtype!=torch.float32 for v in state.values()):raise ValueError('Optimizer state must be FP32')
    return masters,new_optimizer,ema,policy
