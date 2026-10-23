"""BCE, independent Local FM, and parent-token PTv3 FM with shared conditions."""
import math
import torch
from torch import nn


def mlp(a,b,c): return nn.Sequential(nn.Linear(a,b),nn.SiLU(),nn.Linear(b,c))


class HierarchicalOccupancyModel(nn.Module):
    def __init__(self,geometry,*,objective='fm',backbone='ptv3',width=64,lidar_channels=384,
                 condition_mode='pyramid_observed',checkpointing=True,patch_sizes=(64,128),
                 child_bridge='none',child_input='geometry_stats',child_feature_width=24,
                 context_strategy='dynamic'):
        super().__init__()
        if objective not in ('fm','bce') or backbone not in ('local','ptv3'): raise ValueError('Unknown model')
        if condition_mode not in ('none','observed','pyramid','pyramid_observed'): raise ValueError('Unknown condition')
        self.geometry=geometry; self.objective=objective; self.condition_mode=condition_mode; self.backbone_kind=backbone
        if context_strategy not in ('dynamic','static'):raise ValueError('Unknown Stage2 context strategy')
        if context_strategy=='static' and backbone!='ptv3':raise ValueError('Static serialized context requires PTv3')
        self.context_strategy=context_strategy
        if child_bridge not in ('none','child_conditioned') or child_input not in ('geometry_stats','position_only'):
            raise ValueError('Unknown child-aware bridge policy')
        self.child_bridge=child_bridge;self.child_input=child_input;self.child_feature_width=int(child_feature_width)
        if self.child_bridge!='none' and self.child_feature_width<=0:raise ValueError('Positive child feature width required')
        self.position=mlp(3,32,32); self.score=mlp(1,16,16)
        fusion_inputs=32+16
        if 'pyramid' in condition_mode:
            self.lidar=mlp(lidar_channels,width,width); fusion_inputs+=width
        if 'observed' in condition_mode:
            self.observed=mlp(geometry.k,32,32); fusion_inputs+=32
        if objective=='fm':
            self.local=mlp(geometry.k,64,width); self.time=mlp(32,64,32); fusion_inputs+=width+32
            self.register_buffer('frequencies',torch.exp(-math.log(10000.)*torch.arange(16)/15),persistent=False)
            if context_strategy=='static':
                fusion_inputs-=width+32
                self.dynamic_fuse=mlp(width+32,width,width)
        self.fuse=mlp(fusion_inputs,width,width)
        if backbone=='ptv3':
            from .ptv3_backbone import ParentPTv3
            self.trunk=ParentPTv3(width,checkpointing=checkpointing,patch_sizes=patch_sizes,
                                  time_conditioned=objective=='fm' and context_strategy=='dynamic')
        else: self.trunk=nn.Sequential(nn.LayerNorm(width),mlp(width,2*width,width))
        if self.child_bridge=='none':
            decoder_in=(width+width if objective=='fm' and context_strategy=='static'
                        else width+(geometry.k+32 if objective=='fm' else 0))
            self.decoder=nn.Sequential(nn.LayerNorm(decoder_in),nn.Linear(decoder_in,width),nn.SiLU(),nn.Linear(width,geometry.k))
            nn.init.normal_(self.decoder[-1].weight,std=.001); nn.init.zeros_(self.decoder[-1].bias)
        else:
            # One serialized token remains per Parent. A shared scalar decoder
            # reads a distinct local-evidence/position feature for each Child.
            self.child_position=nn.Embedding(geometry.k,self.child_feature_width)
            self.child_evidence=mlp(8,32,self.child_feature_width)
            decoder_in=width+self.child_feature_width+(1+32 if objective=='fm' else 0)
            self.child_decoder=nn.Sequential(nn.LayerNorm(decoder_in),nn.Linear(decoder_in,width),nn.SiLU(),nn.Linear(width,1))
            nn.init.normal_(self.child_decoder[-1].weight,std=.001);nn.init.zeros_(self.child_decoder[-1].bias)

    def decode_children(self,features,state,embedding,batch,valid_child_mask,child_geometry):
        if self.child_bridge=='none':
            decoded=[features]
            if self.objective=='fm':decoded.extend((state.float(),embedding[batch]))
            return (self.decoder(torch.cat(decoded,-1))*valid_child_mask)
        if child_geometry is None or child_geometry.shape!=(len(features),self.geometry.k,8):
            raise ValueError('Child-aware bridge requires input-only [M,K,8] geometry evidence')
        local=child_geometry.float() if self.child_input=='geometry_stats' else torch.zeros_like(child_geometry,dtype=torch.float32)
        child_ids=torch.arange(self.geometry.k,device=features.device)
        child=self.child_position(child_ids)[None].expand(len(features),-1,-1)
        child=child+self.child_evidence(local)
        parts=[features[:,None,:].expand(-1,self.geometry.k,-1),child]
        if self.objective=='fm':
            if state is None or state.shape!=(len(features),self.geometry.k) or embedding is None:
                raise ValueError('FM Child decoder needs [M,K] state and scene-time embedding')
            parts.extend((state.float()[...,None],embedding[batch][:,None,:].expand(-1,self.geometry.k,-1)))
        value=torch.cat(parts,-1)
        return self.child_decoder(value.reshape(-1,value.shape[-1])).reshape(len(features),self.geometry.k)*valid_child_mask

    def forward(self,parent_ids,*,state=None,time=None,lidar_features=None,parent_score=None,observed_child=None,valid_child_mask=None,child_geometry=None):
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
            if self.context_strategy=='dynamic':
                parts.extend((self.local(state.float()),embedding.expand(len(parent_ids),-1)))
        elif state is not None or time is not None: raise ValueError('Direct BCE has neither noise nor time')
        features=self.fuse(torch.cat(parts,-1))
        if self.backbone_kind=='ptv3': features=self.trunk(features,coords,centers,embedding,self.geometry.parent_resolution)
        else: features=features+self.trunk(features)
        if valid_child_mask is None: raise ValueError('Valid-child ROI mask required')
        if self.child_bridge=='none':
            if self.objective=='fm' and self.context_strategy=='static':
                dynamic=self.dynamic_fuse(torch.cat((self.local(state.float()),embedding.expand(len(parent_ids),-1)),-1))
                return self.decoder(torch.cat((features,dynamic),-1))*valid_child_mask
            decoded=[features]
            if self.objective=='fm':decoded.extend((state.float(),embedding.expand(len(parent_ids),-1)))
            return self.decoder(torch.cat(decoded,-1))*valid_child_mask
        batch=torch.zeros(len(parent_ids),device=parent_ids.device,dtype=torch.long)
        return self.decode_children(features,state,embedding,batch,valid_child_mask,child_geometry)
