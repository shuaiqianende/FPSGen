"""FP32 masked FM/shortcut/endpoint, differentiable occupancy and mass losses."""

import torch
import torch.nn.functional as F


def masked_mse(prediction,target,valid):
    error=(prediction.float()-target.float()).square()
    mask=valid.expand_as(error)
    return (error*mask).sum()/mask.sum().clamp_min(1)


def sum_pool(field,factor):
    return F.avg_pool3d(field.float(),factor,stride=factor)*factor**3 if factor!=1 else field.float()


def multiscale_mass_loss(predicted,target,valid,*,weights=(1.,1.,1.,1.)):
    if len(weights)!=4 or any(w<0 for w in weights):
        raise ValueError('Four nonnegative multiscale weights required')
    p,g=predicted.float()*valid,target.float()*valid
    axes=(1,2,3,4)
    pm,gm=p.sum(axes,keepdim=True),g.sum(axes,keepdim=True)
    # During training zero predicted mass must remain a finite differentiable
    # penalty; inference normalization instead rejects it strictly.
    p=p/pm.clamp_min(1e-8);g=g/gm.clamp_min(1e-8)
    values=[(sum_pool(p,f)-sum_pool(g,f)).abs().sum(axes).mean() for f in (1,2,4,8)]
    return sum(w*v for w,v in zip(weights,values)),values


def surface_loss(endpoint,target_counts,valid,mapping,*,multiscale_weight=1.,occupancy_weight=1.,
                 multiscale_weights=(1.,1.,1.,1.),occupancy_temperature=.1):
    if occupancy_temperature<=0 or min(occupancy_weight,multiscale_weight)<0:
        raise ValueError("Positive temperature and nonnegative structural weights required")
    counts=mapping.decode(endpoint)
    threshold=mapping.encode(endpoint.new_tensor(.5))
    logits=(endpoint.float()-threshold)/occupancy_temperature
    target=(target_counts>0)&valid
    mask=valid.expand_as(logits)
    bce=F.binary_cross_entropy_with_logits(logits,target.float(),reduction='none')
    pos=(bce*target).sum()/target.sum().clamp_min(1)
    bg=(bce*((~target)&mask)).sum()/((~target)&mask).sum().clamp_min(1)
    probability=logits.sigmoid()*mask
    dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    occupancy=.5*(pos+bg)+dice
    multi,scales=multiscale_mass_loss(counts,target_counts,valid,weights=multiscale_weights)
    return occupancy_weight*occupancy+multiscale_weight*multi,{'occupancy':occupancy,'multiscale':multi,
        **{f'mass_l1_s{factor}':loss for factor,loss in zip((1,2,4,8),scales)}}
