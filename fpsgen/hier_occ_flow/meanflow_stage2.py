"""Canonical Stage2 endpoint adapter. Fixed geometry is prepared outside JVP.

No vendor source changes, no FlashAttention precision/mask substitution, and no
invented sparse derivative. Unsupported native forward AD must block training.
"""
from contextlib import contextmanager
from dataclasses import dataclass,field
from types import MethodType
import torch
from torch import nn
from .meanflow_interval import install_interval,scene_times


@dataclass
class PreparedStage2:
    parent_ids:torch.Tensor
    batch:torch.Tensor
    order:torch.Tensor
    inverse:torch.Tensor
    coords:torch.Tensor
    centers:torch.Tensor
    valid:torch.Tensor
    static_parts:tuple
    scenes:int
    sparse_indices:torch.Tensor
    topology:object=None
    norm_layouts:dict=field(default_factory=dict)
    pool_layouts:dict=field(default_factory=dict)
    warmed:bool=False


def _norm_forward(module,point):
    mode=getattr(module,'_mf_layout_mode','original')
    if mode=='original':return module._mf_original_forward(point)
    feat=point.feat
    if not len(feat):return point
    if mode=='capture':
        groups,counts=torch.unique_consecutive(point.batch,return_counts=True)
        if len(torch.unique(groups))!=len(groups):raise ValueError('Noncontiguous scene norm layout')
        module._mf_lengths=tuple(counts.tolist())
    lengths=module._mf_lengths
    if lengths is None or sum(lengths)!=len(feat):raise ValueError('Unprepared/mismatched norm layout')
    values=[]
    for scene in feat.split(lengths):
        if len(scene)==1:values.append(scene)
        else:
            source=scene.float() if getattr(module,'_v3_fp32_stats',False) else scene
            centered=source-source.mean(0,keepdim=True)
            values.append((centered*torch.rsqrt(centered.square().mean(0,keepdim=True)+module.eps)).to(feat.dtype))
    normalized=torch.cat(values)
    if module.affine:normalized=normalized*module.weight+module.bias
    point.feat=normalized
    if 'sparse_conv_feat' in point:point.sparse_conv_feat=point.sparse_conv_feat.replace_feature(normalized)
    return point


def _pool_forward(module,point):
    """Same projection/reduction; only immutable sparse metadata is prepared.

    Ordinary source forward is unchanged. Capture uses its exact Point output;
    no warm features, projected features or sparse outputs are retained.
    """
    mode=getattr(module,'_mf_pool_mode','original')
    if mode!='static':
        result=module._mf_original_forward(point)
        if mode=='capture':
            module._mf_pool_layout=dict(sparse_shape=tuple(result.sparse_shape),
                sparse_indices=result.sparse_conv_feat.indices,offset=result.offset)
        return result
    import spconv.pytorch as spconv
    import torch_scatter
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point
    topology=point['_v3_topology'];cached=topology.pooling[id(module)]
    cluster,indices,idx_ptr,head,code,order,inverse,grid,batch,depth,coord=cached
    layout=module._mf_pool_layout
    if layout is None:raise ValueError('Pooling sparse metadata was not prepared')
    data=dict(feat=torch_scatter.segment_csr(module.proj(point.feat)[indices],idx_ptr,reduce=module.reduce),
        coord=coord,grid_coord=grid,serialized_code=code,serialized_order=order,
        serialized_inverse=inverse,serialized_depth=depth,batch=batch,offset=layout['offset'],
        sparse_shape=list(layout['sparse_shape']),_v3_topology=topology,
        _validated_batch_size=point['_validated_batch_size'])
    for key in ('condition','context'):
        if key in point:data[key]=point[key]
    if module.traceable:data.update(pooling_inverse=cluster,pooling_parent=point)
    result=Point(data)
    if module.norm is not None:result=module.norm(result)
    if module.act is not None:result=module.act(result)
    result.sparse_conv_feat=spconv.SparseConvTensor(result.feat,layout['sparse_indices'],
        list(layout['sparse_shape']),point['_validated_batch_size'])
    return result


class MeanFlowStage2(nn.Module):
    def __init__(self,base):
        super().__init__()
        if (base.objective!='fm' or base.context_strategy!='dynamic' or base.child_bridge!='none'
                or base.state_condition_interaction!='none' or base.local_state_encoding!='none'
                or base.geometry.k!=8):raise ValueError('Original dynamic8-child FM only')
        self.base=base
        self.initialization_audit=install_interval(base,'time',32,64)
        self._norms={};self._pools={}
        if base.backbone_kind=='ptv3':
            from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointInstanceNorm,SerializedPooling
            for name,module in base.trunk.named_modules():
                if isinstance(module,PointInstanceNorm):
                    module._mf_original_forward=module.forward
                    module._mf_layout_mode='original';module._mf_lengths=None
                    module.forward=MethodType(_norm_forward,module)
                    self._norms[name]=module
                elif isinstance(module,SerializedPooling):
                    module._mf_original_forward=module.forward
                    module._mf_pool_mode='original';module._mf_pool_layout=None
                    module.forward=MethodType(_pool_forward,module);self._pools[name]=module

    def _apply(self,fn):
        frequency=self.base.frequencies.detach().float().clone()
        result=super()._apply(fn);self.base.frequencies=frequency.to(self.base.frequencies.device)
        return result

    def set_ptv3_precision(self,mode):self.base.set_ptv3_precision(mode)

    @contextmanager
    def _layouts(self,context,mode):
        previous=[]
        for name,module in self._norms.items():
            previous.append((module,module._mf_layout_mode,module._mf_lengths))
            module._mf_layout_mode=mode
            module._mf_lengths=context.norm_layouts.get(name)
        pool_previous=[]
        for name,module in self._pools.items():
            pool_previous.append((module,module._mf_pool_mode,module._mf_pool_layout))
            module._mf_pool_mode=mode;module._mf_pool_layout=context.pool_layouts.get(name)
        try:yield
        finally:
            if mode=='capture':context.pool_layouts={name:module._mf_pool_layout for name,module in self._pools.items()}
            for module,old_mode,layout in pool_previous:module._mf_pool_mode=old_mode;module._mf_pool_layout=layout
            if mode=='capture':context.norm_layouts={name:module._mf_lengths for name,module in self._norms.items()}
            for module,old_mode,lengths in previous:module._mf_layout_mode=old_mode;module._mf_lengths=lengths

    def prepare_conditions(self,parent_ids,*,batch=None,lidar_features,parent_score,observed_child,valid_child_mask):
        if batch is None:batch=torch.zeros_like(parent_ids)
        if not len(parent_ids) or batch.shape!=parent_ids.shape:raise ValueError('Nonempty packed Parents required')
        scenes=int(batch.max())+1
        if not torch.equal(torch.unique(batch),torch.arange(scenes,device=batch.device)):raise ValueError('Contiguous scene IDs required')
        g=self.base.geometry
        if bool(((parent_ids<0)|(parent_ids>=g.num_parents)).any()):raise ValueError('Parent outside geometry')
        keys=batch*g.num_parents+parent_ids;order=torch.argsort(keys);inverse=torch.argsort(order)
        if len(keys)>1 and bool((keys[order][1:]==keys[order][:-1]).any()):raise ValueError('Duplicate Parent within a scene')
        ids=parent_ids[order];b=batch[order];coords=g.unflatten(ids);centers=g.centers(coords).float()
        position=2*(centers-centers.new_tensor(g.origin))/centers.new_tensor([y-x for x,y in zip(g.origin,g.maximum)])-1
        explicit=self.base.ptv3_precision=='explicit_fp16'
        def cast(module,x):return x.to(next(module.parameters()).dtype) if explicit else x.float()
        parts=(self.base.position(cast(self.base.position,position)),
               self.base.score(cast(self.base.score,parent_score[order].reshape(-1,1))),
               self.base.lidar(cast(self.base.lidar,lidar_features[order])),
               self.base.observed(cast(self.base.observed,observed_child[order])))
        # These projections retain their parameter graph and are recreated each
        # optimizer step. They are never detached or cached across updates.
        topology=None
        if self.base.backbone_kind=='ptv3':
            from .ptv3_inference_topology import PTv3InferenceTopology
            topology=PTv3InferenceTopology.build(coords,centers,b,[n+1 for n in g.parent_resolution],
                self.base.trunk.net.order,(max(g.parent_resolution)-1).bit_length(),scene_count=scenes,level='multilevel')
        indices=torch.cat((b[:,None].int(),coords.int()),1).contiguous()
        return PreparedStage2(ids,b,order,inverse,coords,centers,valid_child_mask[order],parts,scenes,indices,topology)

    def _prepared(self,context,state,time,time_end):
        base=self.base;state=state[context.order]
        t,s=scene_times(time,time_end,context.scenes,validate=False)
        angle=t.float()[:,None]*base.frequencies.float()[None]
        tf=torch.cat((angle.sin(),angle.cos()),-1)
        explicit=base.ptv3_precision=='explicit_fp16'
        if explicit:tf=tf.to(next(base.time.parameters()).dtype)
        with base.time.condition(s-t):embedding=base.time(tf)
        local=state.float()
        if explicit:local=local.to(next(base.local.parameters()).dtype)
        parts=[*context.static_parts,base.encode_flow_state(local,context.valid),embedding[context.batch]]
        features=torch.cat(parts,-1)
        if explicit:features=features.to(next(base.fuse.parameters()).dtype)
        features=base.fuse(features)
        if base.backbone_kind=='ptv3':
            import spconv.pytorch as spconv
            net=base.trunk.net
            dtype=next(net.embedding.parameters()).dtype if explicit else torch.float32
            point=context.topology.make_point(features.to(dtype),embedding.to(dtype))
            point['_validated_batch_size']=context.scenes
            with torch.autocast(features.device.type,enabled=False):
                point['sparse_conv_feat']=spconv.SparseConvTensor(point.feat,context.sparse_indices,
                    list(context.topology.sparse_shape),context.scenes)
            if base.ptv3_precision in ('amp_fp16','selective_fp16'):
                with torch.autocast(features.device.type,dtype=torch.float16,enabled=features.is_cuda):
                    point=net.embedding(point);point=net.enc(point);point=net.dec(point)
            else:
                with torch.autocast(features.device.type,enabled=False):
                    point=net.embedding(point);point=net.enc(point);point=net.dec(point)
            features=point.feat
        else:features=features+base.trunk(features)
        if explicit:features=features.to(next(base.decoder.parameters()).dtype)
        return base.decode_children(features,state,embedding,context.batch,context.valid,None)[context.inverse]

    def forward_prepared(self,context,state,time,*,time_end):
        mode='static' if context.warmed else 'original'
        with self._layouts(context,mode):return self._prepared(context,state,time,time_end)

    @torch.no_grad()
    def warm_topology(self,context,state,time,*,time_end):
        with self._layouts(context,'capture'):value=self._prepared(context,state,time,time_end)
        context.warmed=True
        return value

    def forward(self,parent_ids,*,state,time,time_end,batch=None,inference_context=None,**condition):
        scenes=inference_context['scenes'] if inference_context is not None else (int(batch.max())+1 if batch is not None else 1)
        t,s=scene_times(time,time_end,scenes)
        if inference_context is None and self.base.ptv3_precision=='explicit_fp16':
            context=self.prepare_conditions(parent_ids,batch=batch,**condition)
            return self.forward_prepared(context,state,t,time_end=s)
        with self.base.time.condition(s-t):
            return self.base(parent_ids,state=state,time=t,batch=batch,inference_context=inference_context,**condition)

    def prepare_inference_context(self,*args,**kwargs):return self.base.prepare_inference_context(*args,**kwargs)

    def forward_with_inference_context(self,context,state,time,*,time_end):
        t,s=scene_times(time,time_end,context['scenes'])
        with self.base.time.condition(s-t):return self.base.forward_with_inference_context(context,state,t)
