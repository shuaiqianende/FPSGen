"""EMA half-step targets and counted inference; no intermediate count clipping."""

import copy
import torch
from torch import nn
from .shortcut_unet import batch_scalar


class EMA(nn.Module):
    def __init__(self, model, decay=.999):
        super().__init__()
        if not 0<=decay<1:
            raise ValueError("EMA decay must be in[0,1)")
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.decay=decay

    def train(self,mode=True):
        super().train(False);self.model.eval()
        return self

    @torch.no_grad()
    def update(self,model):
        for ema,value in zip(self.model.parameters(),model.parameters()):
            ema.lerp_(value.detach(),1-self.decay)
        for ema,value in zip(self.model.buffers(),model.buffers()):
            ema.copy_(value.detach())


def linear_pair(source,target,t):
    tb=batch_scalar(t,source)[:,None,None,None,None]
    return (1-tb)*source+tb*target,target-source


@torch.no_grad()
def bootstrap_target(ema,state,t,half_h,condition,*,valid=None):
    t,h=batch_scalar(t,state),batch_scalar(half_h,state)
    if bool(((h<=0)|(t+2*h>1.000001)).any()):
        raise ValueError("Bootstrap needs t+2h<=1")
    ema.eval()
    u1=ema(state,t,h,condition)
    if valid is not None:u1=u1*valid
    mid=state+h[:,None,None,None,None]*u1
    u2=ema(mid,t+h,h,condition)
    if valid is not None:u2=u2*valid
    return (.5*(u1+u2)).detach()


@torch.no_grad()
def rollout(model,source,condition,*,nfe,shortcut=True,return_path=False,valid=None):
    if nfe not in (1,2,4,8,16,32):
        raise ValueError("Registered NFE is1/2/4/8/16/32")
    state=source.clone();path=[state.clone()] if return_path else None
    calls=0
    for i in range(nfe):
        t=i/nfe
        velocity=model(state,t,1/nfe if shortcut else 0.,condition,instantaneous=not shortcut)
        if valid is not None:velocity=velocity*valid
        state=state+velocity/nfe;calls+=1
        if not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite rollout state")
        if path is not None:
            path.append(state.clone())
    return state,{'actual_nfe':calls,'path':path}


def path_diagnostics(path):
    if len(path)<3:
        raise ValueError("At least two segments required; one-stepR=1 is not evidence")
    deltas=[(b-a).float().flatten() for a,b in zip(path[:-1],path[1:])]
    chord=(path[-1]-path[0]).float().flatten().norm()
    length=torch.stack([d.norm() for d in deltas]).sum()
    cosines=[torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12) for a,b in zip(deltas[:-1],deltas[1:])]
    return {'path_length_ratio':float(length/chord.clamp_min(1e-12)),
            'mean_velocity_direction_cosine':float(torch.stack(cosines).mean()),'segments':len(deltas)}
