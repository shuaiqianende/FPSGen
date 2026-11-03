"""Explicit FP16 parameters/features with FP32 normalization statistics and master weights.

Controlled adapters only. No vendor edit, global default dtype or coordinate cast.
"""
import copy
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F


def norm32(self,x):
    dtype=x.dtype
    with torch.autocast(x.device.type,enabled=False):
        w=self.weight.float() if self.weight is not None else None
        b=self.bias.float() if self.bias is not None else None
        if isinstance(self,nn.GroupNorm):y=F.group_norm(x.float(),self.num_groups,w,b,self.eps)
        else:y=F.layer_norm(x.float(),self.normalized_shape,w,b,self.eps)
    return y.to(dtype)


def batch_norm32(self,x):
    self._check_input_dim(x);factor=0. if self.momentum is None else self.momentum
    if self.training and self.track_running_stats:
        if self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:factor=1./float(self.num_batches_tracked)
    training=self.training or self.running_mean is None and self.running_var is None
    mean=self.running_mean if not self.training or self.track_running_stats else None
    var=self.running_var if not self.training or self.track_running_stats else None
    dtype=x.dtype
    with torch.autocast(x.device.type,enabled=False):
        y=F.batch_norm(x.float(),mean,var,self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,training,factor,self.eps)
    return y.to(dtype)


def explicit_stage2_forward(self,parent_ids,*,batch=None,state=None,time=None,lidar_features=None,parent_score=None,observed_child=None,valid_child_mask=None,child_geometry=None,inference_context=None):
    if inference_context is not None:
        if parent_ids is not inference_context['input_parent_ids_ref']:raise ValueError('Context changed')
        return self.forward_with_inference_context(inference_context,state,time)
    if self.objective!='fm' or self.context_strategy!='dynamic' or self.child_bridge!='none' or self.state_condition_interaction!='none' or self.local_state_encoding!='none':raise ValueError('Registered original dynamic FM only')
    if batch is None:batch=torch.zeros_like(parent_ids)
    if batch.shape!=parent_ids.shape or not len(parent_ids):raise ValueError('Nonempty packed scenes required')
    scenes=int(batch.max())+1
    if not torch.equal(torch.unique(batch),torch.arange(scenes,device=batch.device)):raise ValueError('Contiguous scene IDs')
    key=batch*self.geometry.num_parents+parent_ids;order=torch.argsort(key);inverse=torch.argsort(order)
    if len(key)>1 and bool((key[order][1:]==key[order][:-1]).any()):raise ValueError('Duplicate Parents')
    ids=parent_ids[order];batch=batch[order]
    if valid_child_mask is None or state is None or state.shape!=(len(ids),self.geometry.k):raise ValueError('Conditions/state required')
    if hasattr(self,'score') and parent_score is None:raise ValueError('Score required for score-enabled model')
    if time is None or time.numel()!=scenes or not bool(((time>=0)&(time<=1)).all()):raise ValueError('One time per scene')
    coords=self.geometry.unflatten(ids);centers=self.geometry.centers(coords).float()
    position=2*(centers-centers.new_tensor(self.geometry.origin))/centers.new_tensor([b-a for a,b in zip(self.geometry.origin,self.geometry.maximum)])-1
    parts=[self.position(position.half())]
    if hasattr(self,'score'):parts.append(self.score(parent_score[order].reshape(-1,1).half()))
    if hasattr(self,'lidar'):
        if lidar_features is None:raise ValueError('Actual pyramid required')
        parts.append(self.lidar(lidar_features[order].half()))
    if hasattr(self,'observed'):
        if observed_child is None:raise ValueError('Observed Mask required')
        parts.append(self.observed(observed_child[order].half()))
    state=state[order];valid=valid_child_mask[order]
    angle=time.float().reshape(scenes,1)*self.frequencies.float()[None]
    embedding=self.time(torch.cat((angle.sin(),angle.cos()),-1).half())
    parts.extend((self.encode_flow_state(state.half(),valid),embedding[batch]))
    features=self.fuse(torch.cat(parts,-1).half())
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point
    net=self.trunk.net;point=Point(dict(feat=features.half(),coord=centers,grid_coord=coords.int(),batch=batch.long(),
        sparse_shape=[n+1 for n in self.geometry.parent_resolution],context=embedding.half()))
    point.serialization(order=net.order,depth=(max(self.geometry.parent_resolution)-1).bit_length(),shuffle_orders=False)
    with torch.autocast(features.device.type,enabled=False):point.sparsify()
    with torch.autocast(features.device.type,dtype=torch.float16,enabled=features.device.type=='cuda'):
        point=net.embedding(point);point=net.enc(point);point=net.dec(point)
    return self.decode_children(point.feat.half(),state,embedding,batch,valid,None)[inverse]


def explicit_decode(self,features,state,embedding,batch,valid,child_geometry):
    if self.child_bridge!='none':raise ValueError('Only original velocity head')
    values=[features.half()]
    if self.objective=='fm':values.extend((state.half(),embedding[batch].half()))
    return self.decoder(torch.cat(values,-1))*valid


def apply_explicit_policy(model):
    # Capture buffers BEFORE half(): upcasting already-rounded time frequencies
    # would not preserve their original mathematics.
    buffers={n:b.detach().clone() for n,b in model.named_buffers() if b.is_floating_point()}
    model.half()
    for name,b in buffers.items():
        owner,_,leaf=name.rpartition('.');setattr(model.get_submodule(owner) if owner else model,leaf,b)
    counts={'batch_norm_fp32_stats':0,'group_layer_norm_fp32_stats':0,'scene_norm_fp32_stats':0}
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.forward=MethodType(batch_norm32,module);counts['batch_norm_fp32_stats']+=1
        elif isinstance(module,(nn.GroupNorm,nn.LayerNorm)):module.forward=MethodType(norm32,module);counts['group_layer_norm_fp32_stats']+=1
        if type(module).__name__=='OccupancyConditionEncoder':module._v6_explicit_fp16=True
        if hasattr(module,'_v3_fp32_stats'):module._v3_fp32_stats=True
        if type(module).__name__=='Stage2OccupancyModel':
            module.set_ptv3_precision('explicit_fp16');module.set_explicit_fp16_norm_policy(True)
            module.forward=MethodType(explicit_stage2_forward,module)
            module.decode_children=MethodType(explicit_decode,module)
        if getattr(module,'_v3_fp32_stats',False):counts['scene_norm_fp32_stats']+=1
    if any(p.dtype!=torch.float16 for p in model.parameters()):raise ValueError('All network parameters must actually be FP16')
    return dict(mode='explicit_fp16_network_fp32_master',parameter_count=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),parameter_dtypes=['torch.float16'],
        fp32_buffers=list(buffers),**counts)


class FP32Masters:
    def __init__(self,model,initial):
        self.params={n:nn.Parameter(initial[n].detach().to(device=p.device,dtype=torch.float32).clone(),requires_grad=True) for n,p in model.named_parameters()}
        self.model=model
    def optimizer_groups(self,groups):
        names={id(p):n for n,p in self.model.named_parameters()};result=[]
        for group in groups:
            g={k:v for k,v in group.items() if k!='params'};g['params']=[self.params[names[id(p)]] for p in group['params']];result.append(g)
        return result
    def copy_gradients(self):
        for name,p in self.model.named_parameters():self.params[name].grad=None if p.grad is None else p.grad.detach().float().clone()
    @torch.no_grad()
    def copy_to_network(self):
        for name,p in self.model.named_parameters():p.copy_(self.params[name].to(p.dtype))
    def state_dict(self):return {n:p.detach().clone() for n,p in self.params.items()}
    @torch.no_grad()
    def load_state_dict(self,state):
        if state.keys()!=self.params.keys():raise ValueError('Master layout changed')
        for n,p in self.params.items():
            if state[n].dtype!=torch.float32 or state[n].shape!=p.shape:raise ValueError('FP32 master shape/dtype mismatch')
            p.copy_(state[n])


@torch.no_grad()
def update_fp32_ema(ema_state,ema_model,model,masters,decay=.999):
    state=model.state_dict()
    for name,dst in ema_state.items():
        source=masters.params[name].detach() if name in masters.params else state[name]
        if dst.is_floating_point():dst.mul_(decay).add_(source.float(),alpha=1-decay)
        else:dst.copy_(source)
    ema_model.load_state_dict(ema_state,strict=True)


def attach_dtype_audit(model):
    records={};handles=[]
    def dtype_of(value):
        if torch.is_tensor(value):return str(value.dtype) if value.is_floating_point() else None
        if isinstance(value,dict):
            if 'feat' in value:return dtype_of(value['feat'])
            for item in value.values():
                dtype=dtype_of(item)
                if dtype:return dtype
        if hasattr(value,'features'):return dtype_of(value.features)
        if isinstance(value,(tuple,list)):
            for item in value:
                dtype=dtype_of(item)
                if dtype:return dtype
        return None
    for name,child in model.named_modules():
        params=list(child.parameters(recurse=False))
        if not params:continue
        def hook(module,args,output,key=name):
            if key not in records:records[key]=dict(type=type(module).__name__,parameter_dtypes=sorted({str(p.dtype) for p in module.parameters(recurse=False)}),input_dtype=dtype_of(args),output_dtype=dtype_of(output))
        handles.append(child.register_forward_hook(hook))
    return records,handles
