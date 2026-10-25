"""Positive-time finite-interval MeanFlow sampling; no JVP at inference."""
import torch


@torch.no_grad()
def sample_meanflow(u_fn,source,valid,*,nfe,scenes,row_to_scene=None,return_velocities=False):
    if nfe not in (1,2,4,8):raise ValueError('Registered NFE1/2/4/8 only')
    mask=valid.bool().expand_as(source);state=source.float()*mask;velocities=[]
    for step in range(nfe):
        time=torch.full((scenes,),step/nfe,device=state.device,dtype=torch.float32)
        end=torch.full_like(time,(step+1)/nfe)
        velocity=u_fn(state,time,end)
        if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('Nonfinite sampled MeanFlow velocity')
        state=(state+(1/nfe)*velocity.float())*mask
        if return_velocities:velocities.append(velocity.detach())
    return (state,velocities) if return_velocities else state


def stage1_score(endpoint):return ((endpoint.float()+1)/2).clamp(0,1)
