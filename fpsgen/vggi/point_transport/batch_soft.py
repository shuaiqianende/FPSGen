"""Scene-isolated CUDA batch soft transport, equal [B,N,3] input clouds.

No loop over scenes in BVH kernels, graph construction or Sinkhorn solves.
Mixed precision changes cost storage only; all geometry and arithmetic FP32.
"""
from functools import lru_cache
from pathlib import Path
import math,time
import torch
from .bvh_neighbors import extension as bvh_extension
from .soft_graph import decode,transpose,extension as graph_extension

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    build=Path(__file__).resolve().parents[1]/'runs/point_transport_batch_v1/cuda_build'
    build.mkdir(parents=True,exist_ok=True)
    return load(name='vggi_batch_soft',sources=[str(Path(__file__).with_suffix('.cu'))],
                build_directory=str(build),extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)

def validate(x):
    if not x.is_cuda or x.dtype!=torch.float32 or x.ndim!=3 or x.shape[2]!=3 or min(x.shape[:2])<1 or x.shape[0]>65535 or x.shape[0]*x.shape[1]>=2**31:
        raise ValueError('Nonempty CUDA FP32 [B,N,3] required; B*N < 2^31')
    if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite coordinates')

class BatchBVH:
    @torch.no_grad()
    def __init__(self,points,leaf_size=512):
        if not isinstance(leaf_size,int) or leaf_size<1:raise ValueError("Positive leaf size required")
        self.points=points.contiguous();b,n,_=points.shape
        depth=min(n.bit_length()-1,max(0,((n+leaf_size-1)//leaf_size-1).bit_length()))
        self.leaves=1<<depth;ext=bvh_extension()
        codes=ext.batch_morton(self.points,self.points.amin(1),self.points.amax(1))
        self.ids=codes.argsort(dim=1)
        self.sorted_points=self.points.gather(1,self.ids[:,:,None].expand(-1,-1,3)).contiguous()
        self.boxes,self.minima=ext.batch_build(self.sorted_points,self.ids,self.leaves)
    @torch.no_grad()
    def query(self,other,k,hint=None):
        if hint is not None:
            b,n,_=other.points.shape
            candidates=self.points.gather(1,hint.reshape(b,-1)[:,:,None].expand(-1,-1,3)).reshape(b,n,k,3)
            d=(other.points[:,:,None,:]-candidates).square()
            bounds=((d[...,0]+d[...,1])+d[...,2]).amax(2)*(1+2e-6)+1e-20
            return bvh_extension().batch_knn_bound(other.sorted_points,other.ids,self.sorted_points,self.ids,self.boxes,self.minima,self.leaves,k,bounds.contiguous())[1]
        return bvh_extension().batch_knn(other.sorted_points,other.ids,self.sorted_points,self.ids,self.boxes,self.minima,self.leaves,k)[1]

@torch.no_grad()
def graph(x,y,xi,yi,k,cost_dtype=torch.float32):
    b,n,_=x.shape;total=b*n
    forward=yi.query(xi,k);reverse=forward if xi is yi else xi.query(yi,k)
    keys=extension().keys(forward,reverse)
    keys=torch.unique(keys,sorted=True)
    s,t,cost=decode(keys,x.reshape(-1,3),y.reshape(-1,3))
    offsets=torch.cat((keys.new_zeros(1),torch.bincount(s,minlength=total).cumsum(0)))
    if xi is yi:
        ot,sc,ct=offsets,t,cost
    else:
        ot=torch.cat((keys.new_zeros(1),torch.bincount(t,minlength=total).cumsum(0)))
        sc,ct=transpose(s,t,cost,ot,total)
    if cost_dtype!=torch.float32:
        cost=cost.to(cost_dtype);ct=cost if xi is yi else ct.to(cost_dtype)
        if not bool(torch.isfinite(cost).all()):raise FloatingPointError('Cost overflow in selected storage precision')
    return dict(keys=keys,s=s,t=t,cost=cost,offsets=offsets,offsets_t=ot,sc=sc,cost_t=ct)

@torch.no_grad()
def solve(x,y,G,eps,penalty,iterations,batch):
    ext=extension();theta=1. if math.isinf(penalty) else penalty/(penalty+eps)
    f=x.new_zeros(len(x));g=torch.zeros_like(f)
    for _ in range(iterations):
        ext.dual(G['offsets'],G['t'],G['cost'],g,f,eps,theta)
        ext.dual(G['offsets_t'],G['sc'],G['cost_t'],f,g,eps,theta)
    ext.dual(G['offsets'],G['t'],G['cost'],g,f,eps,theta)
    src=ext.moments(G['offsets'],G['t'],G['cost'],f,g,x,y,eps)
    dst=ext.moments(G['offsets_t'],G['sc'],G['cost_t'],g,f,y,x,eps)
    fixed=torch.empty_like(g);ext.dual(G['offsets_t'],G['sc'],G['cost_t'],f,fixed,eps,theta)
    r=src[:,0].reshape(batch,-1);c=dst[:,0].reshape(batch,-1)
    safe=r.clamp_min(1e-30);stats=src.reshape(batch,-1,8)
    diagnostics=torch.stack(((r-1).abs().mean(1),(r-1).abs().amax(1),
        (c-1).abs().mean(1),(c-1).abs().amax(1),(c*c.clamp_min(1e-30).log()-c+1).mean(1),
        (fixed-g).abs().reshape(batch,-1).amax(1),stats[:,:,4].mean(1),
        (safe.square()/stats[:,:,6].clamp_min(1e-30)).mean(1),
        (safe.log()-stats[:,:,5]/safe).mean(1),(stats[:,:,7]/safe).mean(1)),1)
    return src,dst,diagnostics

DIAGNOSTICS=('row_mean','row_max','col_mean','col_max','col_kl','fixed_point_max_m2','coupling_cost_half_m2','effective_neighbors_mean','entropy_per_row','max_weight_mean')

@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def match(source,target,*,k=32,epsilons=(.02,.005,.002),penalty=50.,iterations=30,
          step_size=.9,debias=True,cost_precision='fp32',compact_indices=True,graph_backend='sort',leaf_size=512,reuse_knn_bounds=False):
    validate(source);validate(target)
    if source.shape!=target.shape or source.device!=target.device or k not in (32,64) or source.shape[1]<k:
        raise ValueError('Equal scene shapes/devices and N>=K=32/64 required')
    if not epsilons or any(not math.isfinite(e) or e<=0 for e in epsilons) or not penalty>0 or not math.isfinite(step_size) or not 0<step_size<=1 or not isinstance(iterations,int) or iterations<1:
        raise ValueError('Invalid soft solver schedule')
    dtypes={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}
    if cost_precision not in dtypes:raise ValueError('cost_precision must be fp32/fp16/bf16')
    if reuse_knn_bounds and graph_backend!='linear':raise ValueError('Bounds reuse requires linear graph')
    if graph_backend not in ('sort','linear'):raise ValueError('Unknown graph backend')
    build_graph=graph
    if graph_backend=='linear':
        from .linear_graph import graph as build_graph, extension as linear_extension
        if cost_precision!='fp32':raise ValueError('Linear graph requires FP32 costs')
        linear_extension()
    b,n,_=source.shape;device=source.device
    with torch.cuda.device(device):
        bvh_extension();graph_extension();extension()
        torch.cuda.synchronize(device);start=time.perf_counter();z=source.contiguous().clone();y=target.contiguous()
        yi=BatchBVH(y,leaf_size);timings=[];traces=[];cross_hints={};self_hints={}
        for eps in epsilons:
            t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True);t2=torch.cuda.Event(enable_timing=True);t0.record()
            xi=BatchBVH(z,leaf_size);cross=build_graph(z,y,xi,yi,k,dtypes[cost_precision],**({'hints':cross_hints} if reuse_knn_bounds else {}))
            # Only CSR/CSC is needed after graph construction, not audit keys/rows.
            cross.pop('keys');cross.pop('s')
            selfgraph=build_graph(z,z,xi,xi,k,dtypes[cost_precision],**({'hints':self_hints} if reuse_knn_bounds else {})) if debias else None
            if selfgraph is not None:selfgraph.pop('keys');selfgraph.pop('s')
            if compact_indices:
                for G in (cross,selfgraph):
                    if G is None:continue
                    symmetric=G['t'].data_ptr()==G['sc'].data_ptr()
                    G['t']=G['t'].to(torch.int32)
                    G['sc']=G['t'] if symmetric else G['sc'].to(torch.int32)
            t1.record()
            src,_,ci=solve(z.reshape(-1,3),y.reshape(-1,3),cross,eps,penalty,iterations,b)
            direction=-src[:,1:4]
            si=None
            if debias:
                ss,st,si=solve(z.reshape(-1,3),z.reshape(-1,3),selfgraph,eps,penalty,iterations,b)
                direction+=.5*(ss[:,1:4]+st[:,1:4])
            update=step_size*direction/src[:,0,None].clamp_min(.05)
            z=z+update.reshape_as(z);t2.record();timings.append((t0,t1,t2))
            traces.append((ci,si,update.reshape(b,n,3).square().sum(2).mean(1).sqrt()))
            del cross,selfgraph,xi
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite batch soft output')
        trace=[]
        for ci,si,u in traces:
            trace.append(dict(cross=[dict(zip(DIAGNOSTICS,v)) for v in ci.cpu().tolist()],
                self=None if si is None else [dict(zip(DIAGNOSTICS,v)) for v in si.cpu().tolist()],update_rms_m=u.cpu().tolist()))
        torch.cuda.synchronize(device)
        return z,dict(seconds=time.perf_counter()-start,batch=b,points_per_scene=n,k=k,
            graph_seconds=sum(a.elapsed_time(c) for a,c,_ in timings)/1000,
            solve_seconds=sum(c.elapsed_time(d) for _,c,d in timings)/1000,
            trace=trace,reuse_knn_bounds=reuse_knn_bounds,leaf_size=leaf_size,graph_backend=graph_backend,cost_precision=cost_precision,compact_indices=compact_indices,arithmetic_precision='fp32',hard_assignment=False,
            parallel_batch=True,training_label_accepted=False,
            limitation='Scene-isolated sparse finite-iteration soft transport; not converged global OT')
