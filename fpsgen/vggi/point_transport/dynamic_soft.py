"""Only soft couplings and barycentric endpoints; no assignment or rounding.

Objective on a sparse graph (unit source/target masses):
 <C,P> + eps sum P(log P-1) + lambda KL(P^T 1 | 1), P1=1.
 lambda=inf is balanced Sinkhorn, whose sparse support may be infeasible.
 Dynamic intermediate queries expand the graph; C always uses original X.
"""
import math,time
import torch
from .bvh_neighbors import BVHIndex
from .local_neighbors import validate,extension


@torch.no_grad()
def graph(x,y,query,k,previous=None,index_y=None,index_query=None,layout='sorted'):
    if layout not in ('sorted','scatter'):raise ValueError('Invalid graph layout')
    n=len(x);index_y=index_y or BVHIndex(y);qi=index_query or BVHIndex(query)
    forward,_=index_y.query_index(qi,k)
    reverse=forward if index_y is qi else qi.query_index(index_y,k)[0]
    rows=torch.arange(n,device=x.device).repeat_interleave(k)
    keys=torch.cat((rows*n+forward.reshape(-1),reverse.reshape(-1)*n+rows))
    if previous is not None:keys=torch.cat((keys,previous))
    keys=torch.unique(keys,sorted=True)
    if layout=='scatter':
        from .soft_graph import decode
        s,t,cost=decode(keys,x,y)
    else:
        s=keys//n;t=keys%n
        cost=.5*(x[s]-y[t]).square().sum(1)
    offsets=torch.cat((keys.new_zeros(1),torch.bincount(s,minlength=n).cumsum(0)))
    if layout=='scatter' and previous is None and x.data_ptr()==y.data_ptr()==query.data_ptr():
        # The self graph is the exact union of each directed edge and its transpose.
        # Symmetric costs imply CSC is the same CSR, eliminating another sort/copy.
        offsets_t=offsets;sc=t;cost_t=cost
    elif layout=='scatter':
        from .soft_graph import transpose
        offsets_t=torch.cat((keys.new_zeros(1),torch.bincount(t,minlength=n).cumsum(0)))
        sc,cost_t=transpose(s,t,cost,offsets_t,n)
    else:
        perm=t.argsort();sc=s[perm];tc=t[perm]
        offsets_t=torch.cat((keys.new_zeros(1),torch.bincount(tc,minlength=n).cumsum(0)))
        cost_t=cost[perm].contiguous()
    return dict(keys=keys,s=s,t=t,offsets=offsets,offsets_t=offsets_t,sc=sc,
                cost=cost,cost_t=cost_t)


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def solve_graph(x,y,G,*,epsilon=.02,penalty=2.,iterations=40,g=None,relax_source=False,gradients=False,translation_every=0):
    if epsilon<=0 or penalty<=0 or iterations<1:raise ValueError('Invalid solver parameters')
    n=len(x);ext=extension();f=x.new_zeros(n)
    g=x.new_zeros(n) if g is None else g.clone()
    theta=1. if math.isinf(penalty) else penalty/(penalty+epsilon)
    for iteration in range(iterations):
        ext.dual(G['offsets'],G['t'],G['cost'],g,f,epsilon)
        if relax_source:f.mul_(theta)
        ext.dual(G['offsets_t'],G['sc'],G['cost_t'],f,g,epsilon)
        g.mul_(theta)
        if relax_source and math.isfinite(penalty) and translation_every and (iteration+1)%translation_every==0:
            # Exact maximization along f+=shift,g-=shift in the UOT dual.
            # Coupling exponent stays unchanged, while the marginal dual terms improve.
            shift=.5*penalty*(torch.logsumexp(-f/penalty,0)-torch.logsumexp(-g/penalty,0))
            f.add_(shift);g.sub_(shift)
    # Source rows exactly normalized for the returned coupling and endpoints.
    ext.dual(G['offsets'],G['t'],G['cost'],g,f,epsilon)
    if relax_source:f.mul_(theta)
    s,t=G['s'],G['t'];q=torch.exp((f[s]+g[t]-G['cost'])/epsilon)
    row=x.new_zeros(n).scatter_add_(0,s,q)
    if not relax_source:q=q/row[s].clamp_min(1e-30)
    row=x.new_zeros(n).scatter_add_(0,s,q);col=x.new_zeros(n).scatter_add_(0,t,q)
    weighted=torch.zeros_like(x).index_add_(0,s,q[:,None]*y[t]);endpoint=weighted/row[:,None].clamp_min(1e-30)
    normalized=q/row[s].clamp_min(1e-30)
    q2=x.new_zeros(n).scatter_add_(0,s,normalized*normalized)
    maximum=x.new_zeros(n).scatter_reduce_(0,s,normalized,reduce='amax',include_self=True)
    entropy=-(normalized*normalized.clamp_min(1e-30).log()).sum()/n
    variance=(q*(y[t]-endpoint[s]).square().sum(1)).sum()/n
    colkl=(col*col.clamp_min(1e-30).log()-col+1).mean()
    fixed=g.new_empty(n)
    ext.dual(G['offsets_t'],G['sc'],G['cost_t'],f,fixed,epsilon);fixed.mul_(theta)
    info=dict(epsilon=epsilon,penalty=None if math.isinf(penalty) else penalty,iterations=iterations,
              edges=len(s),row_mean=float((row-1).abs().mean()),row_max=float((row-1).abs().max()),
              col_mean=float((col-1).abs().mean()),col_max=float((col-1).abs().max()),col_kl=float(colkl),
              fixed_point_max_m2=float((fixed-g).abs().max()),
              effective_neighbors_mean=float((1/q2.clamp_min(1e-30)).mean()),
              effective_neighbors_median=float((1/q2.clamp_min(1e-30)).median()),
              max_weight_mean=float(maximum.mean()),rows_weight_gt_099=float((maximum>.99).float().mean()),
              entropy_per_row=float(entropy),conditional_variance_m2=float(variance),
              coupling_cost_half_m2=float((q*G['cost']).sum()/n),
              endpoint_rms_m=float((endpoint-x).square().sum(1).mean().sqrt()),
              balanced_converged=bool(max(float((row-1).abs().max()),float((col-1).abs().max()))<.01),
              relaxed_problem=not math.isinf(penalty),relax_source=relax_source,translation_every=translation_every)
    if not bool(torch.isfinite(endpoint).all()):raise FloatingPointError('Nonfinite soft endpoints')
    if gradients:
        incoming=torch.zeros_like(y).index_add_(0,t,q[:,None]*x[s])
        return endpoint,g,info,dict(source_gradient=row[:,None]*x-weighted,
                                    target_gradient=col[:,None]*y-incoming,row_mass=row,
                                    raw_entropy=float((q*(q.clamp_min(1e-30).log()-1)).sum()/n),
                                    row_kl=float((row*row.clamp_min(1e-30).log()-row+1).mean()))
    return endpoint,g,info


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def match(source,target,*,k=32,epsilons=(.1,.02),penalty=2.,iterations=30,
          refresh=False,tau=.5):
    validate(source);validate(target)
    if source.shape!=target.shape or source.device!=target.device or k not in (32,64) or len(source)<k:
        raise ValueError('Equal CUDA FP32 clouds, K32/64 required')
    if not 0<=tau<=1 or not epsilons or any(e<=0 for e in epsilons):raise ValueError('Invalid schedule')
    x,y=source.contiguous(),target.contiguous();extension();torch.cuda.synchronize();start=time.perf_counter()
    yi=BVHIndex(y);G=None;g=None;end=x;traces=[];graph_seconds=0;solve_seconds=0
    for stage,epsilon in enumerate(epsilons):
        if stage==0 or refresh:
            torch.cuda.synchronize();t=time.perf_counter()
            query=((1-tau)*x+tau*end).contiguous()
            G=graph(x,y,query,k,previous=None if G is None else G['keys'],index_y=yi)
            torch.cuda.synchronize();graph_seconds+=time.perf_counter()-t
        torch.cuda.synchronize();t=time.perf_counter()
        end,g,info=solve_graph(x,y,G,epsilon=epsilon,penalty=penalty,iterations=iterations,g=g)
        torch.cuda.synchronize();solve_seconds+=time.perf_counter()-t;traces.append(info)
    torch.cuda.synchronize()
    return end,dict(seconds=time.perf_counter()-start,graph_seconds=graph_seconds,solve_seconds=solve_seconds,
                    k=k,refresh=refresh,tau=tau,trace=traces,hard_assignment=False,
                    endpoint_type='Normalized soft coupling barycenters, no argmax, sampling, permutation or rounding',
                    objective='original source squared distance + entropy + optional target marginal KL',
                    training_label_accepted=False)
