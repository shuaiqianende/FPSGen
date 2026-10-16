"""BCE, independent Local FM, and parent-token PTv3 FM with shared conditions."""
import math
import torch
from torch import nn


def mlp(a,b,c): return nn.Sequential(nn.Linear(a,b),nn.SiLU(),nn.Linear(b,c))


class HierarchicalOccupancyModel(nn.Module):
    def __init__(self,geometry,*,objective='fm',backbone='ptv3',width=64,lidar_channels=384,
                 condition_mode='pyramid_observed',checkpointing=True,patch_sizes=(64,128)):
        super().__init__()
        if objective not in ('fm','bce') or backbone not in ('local','ptv3'): raise ValueError('Unknown model')
        if condition_mode not in ('none','observed','pyramid','pyramid_observed'): raise ValueError('Unknown condition')
        self.geometry=geometry; self.objective=objective; self.condition_mode=condition_mode; self.backbone_kind=backbone
        self.position=mlp(3,32,32); self.score=mlp(1,16,16)
        fusion_inputs=32+16
        if 'pyramid' in condition_mode:
            self.lidar=mlp(lidar_channels,width,width); fusion_inputs+=width
        if 'observed' in condition_mode:
            self.observed=mlp(geometry.k,32,32); fusion_inputs+=32
        if objective=='fm':
            self.local=mlp(geometry.k,64,width); self.time=mlp(32,64,32); fusion_inputs+=width+32
            self.register_buffer('frequencies',torch.exp(-math.log(10000.)*torch.arange(16)/15),persistent=False)
        self.fuse=mlp(fusion_inputs,width,width)
        if backbone=='ptv3':
            from .ptv3_backbone import ParentPTv3
            self.trunk=ParentPTv3(width,checkpointing=checkpointing,patch_sizes=patch_sizes,time_conditioned=objective=='fm')
        else: self.trunk=nn.Sequential(nn.LayerNorm(width),mlp(width,2*width,width))
        decoder_in=width+(geometry.k+32 if objective=='fm' else 0)
        self.decoder=nn.Sequential(nn.LayerNorm(decoder_in),nn.Linear(decoder_in,width),nn.SiLU(),nn.Linear(width,geometry.k))
        nn.init.normal_(self.decoder[-1].weight,std=.001); nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,parent_ids,*,state=None,time=None,lidar_features=None,parent_score=None,observed_child=None,valid_child_mask=None):
        coords=self.geometry.unflatten(parent_ids)
        centers=self.geometry.centers(coords).float()
        position=2*(centers-centers.new_tensor(self.geometry.origin))/centers.new_tensor([b-a for a,b in zip(self.geometry.origin,self.geometry.maximum)])-1
        if parent_score is None: raise ValueError('Explicit candidate score required')
        parts=[self.position(position),self.score(parent_score.float().reshape(-1,1))]
        if hasattr(self,'lidar'):
            if lidar_features is None: raise ValueError('LiDAR pyramid features required; no uniform fallback')
            parts.append(self.lidar(lidar_features.float()))
        if hasattr(self,'observed'):
            if observed_child is None: raise ValueError('Input-only observed child mask required')
            parts.append(self.observed(observed_child.float()))
        embedding=None
        if self.objective=='fm':
            if state is None or state.shape!=(len(parent_ids),self.geometry.k): raise ValueError('FM state must be [M,K]')
            if time is None or time.numel()!=1 or not 0<=float(time)<=1: raise ValueError('One common t in [0,1] per scene')
            angle=time.float().reshape(1,1)*self.frequencies[None]
            embedding=self.time(torch.cat((angle.sin(),angle.cos()),-1))
            parts.extend((self.local(state.float()),embedding.expand(len(parent_ids),-1)))
        elif state is not None or time is not None: raise ValueError('Direct BCE has neither noise nor time')
        features=self.fuse(torch.cat(parts,-1))
        if self.backbone_kind=='ptv3': features=self.trunk(features,coords,centers,embedding,self.geometry.parent_resolution)
        else: features=features+self.trunk(features)
        decoded=[features]
        if self.objective=='fm': decoded.extend((state.float(),embedding.expand(len(parent_ids),-1)))
        result=self.decoder(torch.cat(decoded,-1))
        if valid_child_mask is None: raise ValueError('Valid-child ROI mask required')
        return result*valid_child_mask
