"""Read-only rollout helpers; diagnostic noise is keyed by Scene/Parent/Child."""
import hashlib
from contextlib import contextmanager

import numpy as np
import torch


def parent_noise(parent_ids, children, scene_id, seed):
    ids=np.asarray(parent_ids,dtype=np.uint64)
    base=np.uint64(int.from_bytes(hashlib.sha256((str(scene_id)+':'+str(seed)).encode()).digest()[:8],'little'))
    counter=ids[:,None]*np.uint64(children)+np.arange(children,dtype=np.uint64)[None,:]
    def mix(value):
        with np.errstate(over='ignore'):
            z=value+np.uint64(0x9E3779B97F4A7C15)
            z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
            z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        return z^(z>>np.uint64(31))
    u=((mix(counter^base)>>np.uint64(11)).astype(np.float64)+.5)/2**53
    v=((mix(counter^base^np.uint64(0xD1B54A32D192ED03))>>np.uint64(11)).astype(np.float64)+.5)/2**53
    return (np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)).astype(np.float32)


def vector_stats(value):
    v=value.detach().float()
    norm=torch.linalg.vector_norm(v,dim=-1)
    return {'rms':float(v.square().mean().sqrt()),'norm_quantiles':torch.quantile(norm,torch.tensor([.5,.9,.99,1.],device=norm.device)).cpu().tolist(),
            'finite':bool(torch.isfinite(v).all())}


def error_stats(reference,candidate):
    d=candidate.float()-reference.float()
    return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),
            'relative_l2':float(d.norm()/reference.float().norm().clamp_min(1e-12)),
            'sign_flips':int(((candidate>0)!=(reference>0)).sum())}


def rollout(model,ids,condition,source,nfe,*,keep_states=False):
    if model.training:raise ValueError('Diagnostic rollout requires eval')
    if model.objective=='bce':
        endpoint=model(ids,**condition)
        return endpoint,[],[]
    x=source.clone()
    context=(model.prepare_inference_context(ids,**condition) if model.context_strategy=='static' else None)
    stats=[];states=[]
    for step in range(nfe):
        if keep_states:states.append(x.detach().clone())
        t=x.new_tensor(step/nfe)
        velocity=(model(ids,state=x,time=t,**condition) if context is None else
                  model(ids,state=x,time=t,inference_context=context))
        if not torch.isfinite(velocity).all():raise ValueError('Nonfinite diagnostic velocity')
        updated=x+velocity.float()/nfe
        stats.append({'step':step,'time':step/nfe,'velocity':vector_stats(velocity),
                      'update':vector_stats(updated-x)})
        x=updated
    return x,stats,states


@contextmanager
def block_outputs(model):
    outputs={};handles=[]
    wanted=('enc.enc0.block0','enc.enc0.block1','enc.enc1.block0','enc.enc1.block1',
            'dec.dec0.block0','dec.dec0.block1')
    for name,module in model.trunk.net.named_modules():
        if name in wanted:
            def capture(module,args,output,name=name):
                outputs[name]=output.feat.detach().clone()
            handles.append(module.register_forward_hook(capture))
    try:
        yield outputs
    finally:
        for handle in handles:handle.remove()
