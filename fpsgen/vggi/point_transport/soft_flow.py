"""Sparse debiased entropic transport coordinate flow. NO hard matching.

One objective F(X,Y)-.5F(X,X)-.5F(Y,Y), with symmetric row/column KL
relaxation when penalty is finite. Y-only term has zero coordinate gradient.
Finite-iteration sparse solves give approximate envelope gradients, reported.
"""
import time
import torch
from .dynamic_soft import graph,solve_graph
from .local_neighbors import validate,extension
from .bvh_neighbors import BVHIndex


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def reference_match(source,target,*,k=32,epsilons=(.01,.002),penalty=2.,iterations=30,
          step_size=.9,debias=True,translation_every=0,graph_backend='fast',cost_precision='fp32'):
    if source.ndim==3 or target.ndim==3:
        if graph_backend!='fast' or translation_every!=0:
            raise ValueError('Parallel batch supports fast backend and translation_every=0')
        from .batch_soft import match as batch_match
        return batch_match(source,target,k=k,epsilons=epsilons,penalty=penalty,
            iterations=iterations,step_size=step_size,debias=debias,cost_precision=cost_precision)
    if cost_precision!='fp32':raise ValueError('Mixed cost storage requires [B,N,3] input (use B=1 for a single scene)')
    validate(source);validate(target)
    if source.shape!=target.shape or source.device!=target.device or not 0<step_size<=1 or not epsilons:
        raise ValueError('Invalid soft flow inputs')
    if graph_backend not in ('legacy','shared','fast'):raise ValueError('Invalid graph backend')
    if graph_backend=='fast':
        from .soft_graph import extension as graph_extension
        graph_extension()
    z=source.contiguous().clone();y=target.contiguous();extension();torch.cuda.synchronize();start=time.perf_counter()
    yi=BVHIndex(y);trace=[];graph_seconds=0;solve_seconds=0
    for eps in epsilons:
        torch.cuda.synchronize();t=time.perf_counter()
        if graph_backend=='legacy':
            cross=graph(z,y,z,k,index_y=yi)
            self_graph=graph(z,z,z,k) if debias else None
        else:
            zi=BVHIndex(z);layout='scatter' if graph_backend=='fast' else 'sorted'
            cross=graph(z,y,z,k,index_y=yi,index_query=zi,layout=layout)
            self_graph=graph(z,z,z,k,index_y=zi,index_query=zi,layout=layout) if debias else None
        torch.cuda.synchronize();graph_seconds+=time.perf_counter()-t;t=time.perf_counter()
        _,_,ci,cg=solve_graph(z,y,cross,epsilon=eps,penalty=penalty,iterations=iterations,
                             relax_source=True,gradients=True,translation_every=translation_every)
        direction=-cg['source_gradient']
        si=None
        if debias:
            _,_,si,sg=solve_graph(z,z,self_graph,epsilon=eps,penalty=penalty,iterations=iterations,
                                 relax_source=True,gradients=True,translation_every=translation_every)
            direction+=.5*(sg['source_gradient']+sg['target_gradient'])
        # Positive diagonal preconditioner; avoids freezing low cross-mass rows.
        update=step_size*direction/cg['row_mass'][:,None].clamp_min(.05)
        z=z+update
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite soft flow')
        torch.cuda.synchronize();solve_seconds+=time.perf_counter()-t
        trace.append(dict(cross=ci,self=si,update_rms_m=float(update.square().sum(1).mean().sqrt()),
                          rows_mass_below_005=float((cg['row_mass']<.05).float().mean())))
    return z,dict(seconds=time.perf_counter()-start,graph_seconds=graph_seconds,solve_seconds=solve_seconds,
                  trace=trace,k=k,step_size=step_size,debias=debias,graph_backend=graph_backend,hard_assignment=False,
                  endpoint_type='Continuous soft transport gradient steps; no permutation, argmax, NN projection or sampling',
                  objective='F(X,Y) - 0.5 F(X,X) - 0.5 F(Y,Y)' if debias else 'F(X,Y)',
                  limitation='Local graph and finite iterations approximate the gradient; no global convergence guarantee',
                  training_label_accepted=False)


def match(source,target,*,k=32,epsilons=(.02,.005,.002),penalty=50.,iterations=30,
          step_size=.9,debias=True,translation_every=0,graph_backend='optimized',cost_precision='fp32'):
    """Recommended scene-isolated soft matcher for [N,3] or [B,N,3].

    Defaults fixed by the three-scene 180K audit. Explicit legacy backends are
    retained solely for reproducible experiments; no hard matching fallback.
    """
    if graph_backend!='optimized':
        return reference_match(source,target,k=k,epsilons=epsilons,penalty=penalty,iterations=iterations,
            step_size=step_size,debias=debias,translation_every=translation_every,graph_backend=graph_backend,cost_precision=cost_precision)
    if translation_every!=0 or cost_precision!='fp32':
        raise ValueError('Recommended matcher requires FP32 costs and translation_every=0')
    from .batch_soft import match as batch_match
    single=source.ndim==2 and target.ndim==2
    z,info=batch_match(source[None] if single else source,target[None] if single else target,
        k=k,epsilons=epsilons,penalty=penalty,iterations=iterations,step_size=step_size,debias=debias,
        graph_backend='linear',reuse_knn_bounds=True)
    info['implementation']='exact_knn_bounds_linear_union_fp32_v1'
    return (z[0] if single else z),info
