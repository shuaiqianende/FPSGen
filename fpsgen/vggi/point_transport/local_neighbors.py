"""Exact CUDA grid kNN and experimental local transport/DCD targets.

All fine points remain present. Grid shells expand until a distance certificate
is met, otherwise a full scan is used. This certifies the kNN, not sparse OT.
"""
from functools import lru_cache
from pathlib import Path
import math
import time
import torch


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    build=Path(__file__).resolve().parents[1]/'runs/point_transport_local_dcd_v1/cuda_build'
    build.mkdir(parents=True,exist_ok=True)
    return load(name='vggi_grid_neighbors',sources=[str(Path(__file__).with_suffix('.cu'))],
                build_directory=str(build),extra_cuda_cflags=['-O3'],verbose=False)


def validate(x):
    if not x.is_cuda or x.dtype!=torch.float32 or x.ndim!=2 or x.shape[1]!=3 or not len(x):
        raise ValueError('Nonempty CUDA FP32 Nx3 required')
    if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite points')


class GridIndex:
    @torch.no_grad()
    def __init__(self,points,cell=.75):
        validate(points)
        if not math.isfinite(cell) or cell<=0:raise ValueError('positive cell required')
        self.points=points.detach().contiguous();self.cell=float(cell)
        self.origin=(torch.floor(points.amin(0)/cell)*cell).contiguous()
        coords=torch.floor((points-self.origin)/cell).long()
        self.shape=tuple((coords.amax(0)+1).tolist())
        nx,ny,nz=self.shape;cells=nx*ny*nz
        if cells>20_000_000:raise ValueError('Grid too large; increase cell size')
        keys=(coords[:,2]*ny+coords[:,1])*nx+coords[:,0]
        self.order=torch.argsort(keys)
        self.sorted_points=self.points[self.order].contiguous()
        self.offsets=torch.cat((keys.new_zeros(1),torch.bincount(keys,minlength=cells).cumsum(0)))

    @torch.no_grad()
    def query(self,queries,k=1,max_shell=8):
        validate(queries)
        if queries.device!=self.points.device or k not in (1,32,64) or k>len(self.points) or max_shell<0:
            raise ValueError('Invalid query device/K/shells')
        n=len(queries)
        ids=torch.empty((n,k),device=queries.device,dtype=torch.int64)
        dist=torch.empty((n,k),device=queries.device,dtype=torch.float32)
        shells=torch.empty(n,device=queries.device,dtype=torch.int32);fallback=torch.empty_like(shells)
        extension().knn(queries.contiguous(),self.sorted_points,self.order,self.offsets,self.origin,
            *self.shape,self.cell,max_shell,ids,dist,shells,fallback)
        return ids,dist,dict(shells=shells,fallback=fallback)


def dcd_loss(z,y,forward,reverse,alpha=100.):
    """Equal-cardinality DCD; nearest IDs/counts fixed within each gradient step."""
    n=len(z)
    if n!=len(y):raise ValueError('equal clouds required for this DCD')
    f=forward.reshape(-1);r=reverse.reshape(-1)
    cf=torch.bincount(f,minlength=n)[f].detach().float()
    cr=torch.bincount(r,minlength=n)[r].detach().float()
    df=(z-y[f]).square().sum(1);dr=(z[r]-y).square().sum(1)
    return .5*((1-torch.exp(-alpha*df)/(cf+1e-6)).mean()+
               (1-torch.exp(-alpha*dr)/(cr+1e-6)).mean())


@torch.enable_grad()
@torch.cuda.amp.autocast(enabled=False)
def refine_dcd(source,target,*,start=None,steps=8,penalty=.01,cell=.75,step_size=.5,mode='anneal'):
    validate(source);validate(target)
    if source.shape!=target.shape or source.device!=target.device:raise ValueError('equal clouds required')
    if steps<1 or penalty<0 or mode not in ('anneal','fine'):raise ValueError('Invalid optimizer settings')
    anchor=source.detach();y=target.detach();z=(anchor if start is None else start.detach()).clone()
    index_y=GridIndex(y,cell);trace=[];fallback_total=0
    for step in range(steps):
        alpha=(1.,4.,16.,100.)[min(3,step*4//steps)] if mode=='anneal' else 100.
        f,_,af=index_y.query(z);r,_,ar=GridIndex(z,cell).query(y)
        z=z.detach().requires_grad_(True)
        data=dcd_loss(z,y,f,r,alpha)
        move=(z-anchor).square().sum(1).mean()
        loss=data+penalty*move
        grad,=torch.autograd.grad(loss,z)
        update=-(step_size/alpha)*len(z)*grad
        # Trust region prevents high NN multiplicities from causing huge jumps.
        update=update*torch.clamp(.25/update.norm(dim=1,keepdim=True).clamp_min(1e-12),max=1.)
        z=(z+update).detach()
        fallback_total+=int(af['fallback'].sum()+ar['fallback'].sum())
        trace.append(dict(alpha=alpha,dcd=float(data.detach()),displacement_penalty=float(move.detach())))
    if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite DCD endpoints')
    return z,dict(steps=steps,penalty=penalty,step_size=step_size,max_step_m=.25,
                  mode=mode,trace=trace,knn_full_scan_queries=fallback_total,
                  endpoint_type='DCD-optimized source positions; approximate GT distribution')


@torch.no_grad()
def sparse_transport(source,target,*,k=32,iterations=80,cell=.75,tolerance=.01,neighbor_backend='grid'):
    """Bidirectional exact kNN graph, balanced log Sinkhorn, barycentric output.

    Residuals are reported even on failure. Local support need not be feasible;
    an unconverged endpoint is a diagnostic, never an approved training label.
    """
    validate(source);validate(target)
    n=len(source)
    if source.shape!=target.shape or source.device!=target.device:raise ValueError('equal clouds required')
    if iterations<1:raise ValueError('positive iterations required')
    torch.cuda.synchronize();stamp=time.perf_counter()
    if neighbor_backend=='grid':
        forward,_,a=GridIndex(target,cell).query(source,k)
        reverse,_,b=GridIndex(source,cell).query(target,k)
        fallback_count=int(a['fallback'].sum()+b['fallback'].sum())
    elif neighbor_backend=='bvh':
        from .bvh_neighbors import bidirectional_knn
        (forward,_),(reverse,_)=bidirectional_knn(source,target,k)
        fallback_count=0
    else:raise ValueError('neighbor_backend must be grid or bvh')
    torch.cuda.synchronize();knn_seconds=time.perf_counter()-stamp;stamp=time.perf_counter()
    rows=torch.arange(n,device=source.device).repeat_interleave(k)
    keys=torch.cat((rows*n+forward.reshape(-1),reverse.reshape(-1)*n+rows))
    keys=torch.unique(keys,sorted=True)
    s=keys//n;t=keys%n
    offsets=torch.cat((keys.new_zeros(1),torch.bincount(s,minlength=n).cumsum(0)))
    perm=torch.argsort(t);tc=t[perm];sc=s[perm]
    offsets_t=torch.cat((keys.new_zeros(1),torch.bincount(tc,minlength=n).cumsum(0)))
    cost=.5*(source[s]-target[t]).square().sum(1);cost_t=cost[perm].contiguous()
    f=source.new_zeros(n);g=source.new_zeros(n);ext=extension()
    torch.cuda.synchronize();graph_seconds=time.perf_counter()-stamp;stamp=time.perf_counter()
    trace=[];q=None
    for eps in (.5,.15,.05,.01):
        for _ in range(iterations):
            ext.dual(offsets,t,cost,g,f,eps)
            ext.dual(offsets_t,sc,cost_t,f,g,eps)
        q=torch.exp((f[s]+g[t]-cost)/eps)
        row=source.new_zeros(n).scatter_add_(0,s,q);col=source.new_zeros(n).scatter_add_(0,t,q)
        trace.append(dict(epsilon=eps,row_mean=float((row-1).abs().mean()),
            row_max=float((row-1).abs().max()),col_mean=float((col-1).abs().mean()),col_max=float((col-1).abs().max())))
    endpoint=torch.zeros_like(source).index_add_(0,s,q[:,None]*target[t])/row[:,None].clamp_min(1e-30)
    finite=bool(torch.isfinite(endpoint).all() & torch.isfinite(q).all())
    if not finite:raise FloatingPointError('Nonfinite sparse Sinkhorn result')
    torch.cuda.synchronize();solve_seconds=time.perf_counter()-stamp
    return endpoint,dict(k=k,neighbor_backend=neighbor_backend,edges=len(keys),iterations_per_scale=iterations,trace=trace,
        phase_seconds=dict(knn=knn_seconds,graph=graph_seconds,solve_and_output=solve_seconds),
        max_row_degree=int((offsets[1:]-offsets[:-1]).max()),max_col_degree=int((offsets_t[1:]-offsets_t[:-1]).max()),
        balanced_converged=max(trace[-1]['row_max'],trace[-1]['col_max'])<=tolerance,
        tolerance=tolerance,knn_full_scan_queries=fallback_count,
        endpoint_type='sparse coupling barycenters; not a permutation; inspect convergence')


def adaptive_sparse_transport(source,target,*,iterations=80,cell=.75):
    attempts=[]
    for k in (32,64):
        endpoint,info=sparse_transport(source,target,k=k,iterations=iterations,cell=cell)
        attempts.append(info)
        if info['balanced_converged']:break
    return endpoint,dict(attempts=attempts,balanced_converged=info['balanced_converged'],
        endpoint_type='adaptive sparse coupling barycenters',
        training_label_accepted=info['balanced_converged'])
