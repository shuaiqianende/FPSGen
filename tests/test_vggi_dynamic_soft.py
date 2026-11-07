import math
import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from fpsgen.vggi.point_transport.dynamic_soft import graph,solve_graph,match

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

def test_dense_independent_soft_solution():
    torch.manual_seed(51);x=torch.randn(32,3,device='cuda')*.3;y=torch.randn_like(x)*.3
    G=graph(x,y,x,32);eps=.1;lam=.2
    z,g,info=solve_graph(x,y,G,epsilon=eps,penalty=lam,iterations=250)
    xx=x.cpu().numpy().astype(float);yy=y.cpu().numpy().astype(float)
    C=.5*((xx[:,None]-yy[None])**2).sum(2);v=np.zeros(32)
    for _ in range(1000):
        u=-eps*logsumexp((v[None]-C)/eps,axis=1)
        v=-eps*lam/(lam+eps)*logsumexp((u[:,None]-C)/eps,axis=0)
    u=-eps*logsumexp((v[None]-C)/eps,axis=1)
    q=np.exp((u[:,None]+v[None]-C)/eps)
    np.testing.assert_allclose(z.cpu(),q@yy,atol=3e-6,rtol=3e-5)
    assert info['row_max']<1e-5 and info['effective_neighbors_median']>1.1
    assert info['fixed_point_max_m2']<1e-5

def test_balanced_soft_is_not_a_permutation():
    x=torch.zeros(32,3,device='cuda');y=torch.zeros_like(x);y[:,0]=torch.linspace(-1,1,32,device='cuda')
    G=graph(x,y,x,32);z,_,info=solve_graph(x,y,G,epsilon=.2,penalty=math.inf,iterations=50)
    assert z.abs().max()<1e-6 and info['balanced_converged']
    assert info['effective_neighbors_median']>31.9
    assert float(torch.cdist(z,y).min())>.01

def test_graph_expansion_keeps_edges_and_original_cost():
    torch.manual_seed(55);x=torch.randn(75,3,device='cuda');y=torch.randn_like(x)
    a=graph(x,y,x,32);b=graph(x,y,(x+y)*.5,32,previous=a['keys'])
    assert torch.isin(a['keys'],b['keys']).all()
    torch.testing.assert_close(b['cost'],.5*(x[b['s']]-y[b['t']]).square().sum(1))
    with torch.cuda.amp.autocast():z,info=match(x,y,k=32,refresh=True,iterations=10)
    assert z.shape==x.shape and z.dtype==torch.float32 and torch.isfinite(z).all()
    assert not info['hard_assignment'] and len(info['trace'])==2

def test_debiased_gradient_independent_finite_difference():
    from scipy.special import logsumexp
    rng=np.random.default_rng(19);a=rng.normal(size=(32,3))*.2;b=rng.normal(size=(32,3))*.2
    eps=.1;lam=.3;theta=lam/(lam+eps)
    def value(x,y):
        C=.5*((x[:,None]-y[None])**2).sum(2);g=np.zeros(32)
        for _ in range(200):
            f=-eps*theta*logsumexp((g[None]-C)/eps,axis=1)
            g=-eps*theta*logsumexp((f[:,None]-C)/eps,axis=0)
        P=np.exp((f[:,None]+g[None]-C)/eps);r=P.sum(1);c=P.sum(0)
        return ((P*C).sum()+eps*(P*(np.log(P)-1)).sum()+lam*((r*np.log(r)-r+1).sum()+(c*np.log(c)-c+1).sum()))/32
    x=torch.tensor(a,dtype=torch.float32,device='cuda');y=torch.tensor(b,dtype=torch.float32,device='cuda')
    _,_,_,cross=solve_graph(x,y,graph(x,y,x,32),epsilon=eps,penalty=lam,iterations=200,relax_source=True,gradients=True,translation_every=5)
    _,_,_,self_g=solve_graph(x,x,graph(x,x,x,32),epsilon=eps,penalty=lam,iterations=200,relax_source=True,gradients=True)
    grad=(cross['source_gradient']-.5*(self_g['source_gradient']+self_g['target_gradient']))/32
    for i,d in [(0,0),(3,1),(17,2)]:
        plus=a.copy();minus=a.copy();plus[i,d]+=1e-4;minus[i,d]-=1e-4
        fd=((value(plus,b)-.5*value(plus,plus))-(value(minus,b)-.5*value(minus,minus)))/2e-4
        assert abs(float(grad[i,d])-fd)<2e-6

def test_soft_flow_identity_and_no_hard_output():
    from fpsgen.vggi.point_transport.soft_flow import match as flow
    torch.manual_seed(38);x=torch.randn(32,3,device='cuda')*.2
    z,info=flow(x,x,k=32,epsilons=(.1,),penalty=.3,iterations=200)
    torch.testing.assert_close(z,x,atol=2e-6,rtol=2e-5)
    assert not info['hard_assignment']
