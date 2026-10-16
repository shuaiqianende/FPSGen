"""Only straight-path Gaussian conditional FM MSE; no structural auxiliaries."""
import torch
import torch.nn.functional as F


def masked_mean(value,mask):
    if value.shape!=mask.shape or not bool(mask.any()): raise ValueError('Valid child mask mismatch/empty')
    return (value.float()*mask).sum()/mask.sum()


def training_pair(occupancy,valid_mask,generator,t=None):
    if generator is None: raise ValueError('Explicit source RNG required')
    if bool(((occupancy!=0)&(occupancy!=1)).any()): raise ValueError('Binary occupancy required')
    x1=(2*occupancy.float()-1)*valid_mask
    x0=torch.randn(x1.shape,generator=generator,device=x1.device)*valid_mask
    if t is None: t=torch.rand((),generator=generator,device=x1.device)
    if t.numel()!=1 or not 0<=float(t)<=1: raise ValueError('One scalar flow time required')
    return {'source':x0,'target':x1,'time':t,'state':(1-t)*x0+t*x1,'velocity':x1-x0}


def fm_loss(predicted_velocity,target_velocity,valid_mask):
    return masked_mean((predicted_velocity.float()-target_velocity.float()).square(),valid_mask)


def bce_loss(logits,occupancy,valid_mask):
    return masked_mean(F.binary_cross_entropy_with_logits(logits.float(),occupancy.float(),reduction='none'),valid_mask)
