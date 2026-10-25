"""Original MeanFlow with forward noise0->data1 time and fixed endpoint s."""
import torch
from .meanflow_jvp import endpoint_jvp


def sample_endpoints(time,generator,diagonal_probability=.75):
    if generator is None or diagonal_probability!=.75:raise ValueError('Registered independent interval RNG /75% diagonal required')
    end=time+(1-time)*torch.rand(time.shape,device=time.device,dtype=time.dtype,generator=generator)
    diagonal=torch.rand(time.shape,device=time.device,generator=generator)<diagonal_probability
    return torch.where(diagonal,time,end),diagonal


def broadcast_scene(value,state,row_to_scene=None):
    if value.ndim!=1:raise ValueError('Scene scalar vector required')
    if row_to_scene is None:
        if len(value)!=len(state):raise ValueError('Dense scene count mismatch')
        rows=value
    else:rows=value[row_to_scene]
    return rows.reshape(-1,*([1]*(state.ndim-1)))


def scene_squared_errors(prediction,target,valid,row_to_scene=None):
    mask=valid.bool().expand_as(prediction)
    error=(prediction.float()-target.float()).square()*mask
    if row_to_scene is None:
        counts=mask.flatten(1).sum(1)
        if bool((counts==0).any()):raise ValueError('Empty valid scene')
        return error.flatten(1).sum(1)/counts
    values=[]
    for scene in range(int(row_to_scene.max())+1):
        selected=row_to_scene==scene;denominator=mask[selected].sum()
        if not bool(denominator):raise ValueError('Empty valid scene')
        values.append(error[selected].sum()/denominator)
    return torch.stack(values)


def meanflow_loss(u_fn,state,time,time_end,velocity,valid,*,row_to_scene=None,
                  model=None,adaptive_power=0.,adaptive_epsilon=1e-3,skip_all_diagonal=True):
    if state.shape!=velocity.shape:raise ValueError('State/velocity mismatch')
    if time.shape!=time_end.shape or not bool(((time>=0)&(time_end>=time)&(time_end<=1)).all()):
        raise ValueError('Invalid endpoint times')
    mask=valid.bool().expand_as(state);state=state*mask;velocity=velocity*mask
    duration=time_end-time
    if skip_all_diagonal and bool((duration==0).all()):
        derivative=torch.zeros_like(velocity);jvp_calls=0
    else:
        _,derivative=endpoint_jvp(u_fn,state,time,time_end,velocity,model=model)
        derivative=derivative*mask;jvp_calls=1
    target=(velocity+broadcast_scene(duration,state,row_to_scene)*derivative).detach()*mask
    if not bool(torch.isfinite(derivative).all() and torch.isfinite(target).all()):
        raise FloatingPointError('Nonfinite native MeanFlow derivative/target')
    prediction=u_fn(state,time,time_end)*mask
    if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('Nonfinite MeanFlow prediction')
    per_scene=scene_squared_errors(prediction,target,mask,row_to_scene)
    if adaptive_power:
        if adaptive_power<0:raise ValueError('Nonnegative adaptive exponent required')
        loss=((per_scene.detach()+adaptive_epsilon).pow(-adaptive_power)*per_scene).mean()
    elif row_to_scene is None:
        # Identical dense ROI per scene: retain the original FM reduction order.
        loss=((prediction.float()-target.float()).square()*mask).sum()/mask.sum()
    else:loss=per_scene.mean()
    return loss,dict(prediction=prediction,target=target,derivative=derivative,
        per_scene=per_scene,jvp_calls=jvp_calls,diagonal=(duration==0),duration=duration)
