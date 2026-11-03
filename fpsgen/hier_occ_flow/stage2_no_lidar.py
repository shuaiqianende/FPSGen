"""Physically remove LiDAR and Score branches from Stage2 scratch model."""
import torch
from torch import nn
from .stage2_remove_score import remove_score

def remove_lidar_and_score(model,optimizer):
 optimizer,_=remove_score(model,optimizer,model.state_dict())
 if not hasattr(model,'lidar') or model.fuse[0].in_features<96:raise ValueError('Expected 32D position + 64D LiDAR')
 old_names={id(p):n for n,p in model.named_parameters()};old_groups=optimizer.param_groups;old_state=optimizer.state
 if old_state:raise ValueError('Scratch initialization must not contain optimizer state')
 old=model.fuse[0];new=nn.Linear(old.in_features-64,old.out_features,device=old.weight.device,dtype=old.weight.dtype)
 with torch.no_grad():new.weight.copy_(torch.cat((old.weight[:,:32],old.weight[:,96:]),dim=1));new.bias.copy_(old.bias)
 model.fuse[0]=new;del model.lidar;model.condition_mode='observed'
 names=dict(model.named_parameters());groups=[]
 for group in old_groups:
  params=[names[old_names[id(p)]] for p in group['params'] if not old_names[id(p)].startswith('lidar.')]
  groups.append({**{k:v for k,v in group.items() if k!='params'},'params':params})
 return type(optimizer)(groups,**optimizer.defaults)
