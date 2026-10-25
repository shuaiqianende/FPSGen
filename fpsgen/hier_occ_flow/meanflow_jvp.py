"""Exact native forward AD only. Finite differences are diagnostics, not targets."""
from contextlib import contextmanager
import torch


@contextmanager
def without_activation_checkpointing(model):
    flags=[]
    for module in model.modules():
        for name in ('checkpointing','activation_checkpointing'):
            if isinstance(getattr(module,name,None),bool):
                flags.append((module,name,getattr(module,name)));setattr(module,name,False)
    try:yield
    finally:
        for module,name,value in reversed(flags):setattr(module,name,value)


def endpoint_jvp(u_fn,state,time,time_end,velocity,*,model=None):
    """Hold endpoint fixed: tangent(v,1,0). no_grad does not disable forward AD."""
    from contextlib import nullcontext
    context=without_activation_checkpointing(model) if model is not None else nullcontext()
    with context,torch.no_grad():
        primal,derivative=torch.func.jvp(u_fn,(state,time,time_end),
            (velocity,torch.ones_like(time),torch.zeros_like(time_end)))
    return primal.detach(),derivative.detach()


def finite_difference_diagnostic(u_fn,state,time,time_end,velocity,epsilon):
    with torch.no_grad():
        plus=u_fn(state+epsilon*velocity,time+epsilon,time_end)
        minus=u_fn(state-epsilon*velocity,time-epsilon,time_end)
    return (plus-minus)/(2*epsilon)
