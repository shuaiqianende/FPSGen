import pytest
import torch
from fpsgen.vggi.point_transport.batch_soft import BatchBVH,graph,solve,match
from fpsgen.vggi.point_transport.bvh_neighbors import BVHIndex
from fpsgen.vggi.point_transport.dynamic_soft import graph as single_graph,solve_graph
from fpsgen.vggi.point_transport.soft_flow import match as public_match
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('k',[32,64])
def test_segmented_bvh_and_graph(k):
    torch.manual_seed(31);x=torch.randn(3,137,3,device='cuda');y=torch.randn_like(x)
    # Completely overlapping scenes still must use distinct supports.
    x[1]=x[0];y[1]=y[0]+.05
    xi,yi=BatchBVH(x),BatchBVH(y);ids=yi.query(xi,k)
    G=graph(x,y,xi,yi,k)
    assert torch.equal(G['s']//137,G['t']//137)
    for b in range(3):
        one=BVHIndex(y[b]).query_index(BVHIndex(x[b]),k)[0]
        assert torch.equal(ids[b],one)
        # Brute-force FP32 distances independently verify exact neighbors.
        d=(x[b,:,None]-y[b,None,:]).square().sum(2)
        torch.testing.assert_close(d.gather(1,ids[b]),d.sort(1).values[:,:k],rtol=2e-6,atol=2e-6)
        original=single_graph(x[b],y[b],x[b],k)
        selected=G['s']//137==b
        keys=(G['s'][selected]%137)*137+G['t'][selected]%137
        assert torch.equal(keys,original['keys'])
        torch.testing.assert_close(G['cost'][selected],original['cost'],rtol=0,atol=0)
    # Symmetric graph aliases preserve the complete transposed edge set.
    S=graph(x,x,xi,xi,k);assert S['offsets_t'].data_ptr()==S['offsets'].data_ptr()
    cols=torch.arange(411,device='cuda').repeat_interleave(G['offsets_t'][1:]-G['offsets_t'][:-1])
    order=(G['sc']*411+cols).argsort()
    assert torch.equal((G['sc']*411+cols)[order],G['keys'])
    torch.testing.assert_close(G['cost_t'][order],G['cost'],rtol=0,atol=0)

@pytest.mark.parametrize('precision',[torch.float32,torch.float16,torch.bfloat16])
def test_fused_moments_against_unfused_coupling(precision):
    torch.manual_seed(4);x=torch.randn(2,32,3,device='cuda')*.2;y=torch.randn_like(x)*.2
    G=graph(x,y,BatchBVH(x),BatchBVH(y),32,precision)
    src,dst,diag=solve(x.reshape(-1,3),y.reshape(-1,3),G,.1,.3,100,2)
    # Same quantized costs, then evaluate with independent old FP32 implementation.
    ref={**G,'cost':G['cost'].float(),'cost_t':G['cost_t'].float()}
    _,_,info,grad=solve_graph(x.reshape(-1,3),y.reshape(-1,3),ref,epsilon=.1,penalty=.3,iterations=100,relax_source=True,gradients=True)
    torch.testing.assert_close(src[:,0],grad['row_mass'],rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(src[:,1:4],grad['source_gradient'],rtol=2e-4,atol=2e-6)
    torch.testing.assert_close(dst[:,1:4],grad['target_gradient'],rtol=2e-4,atol=2e-6)
    assert torch.isfinite(diag).all()
    compressed={**G,'t':G['t'].to(torch.int32),'sc':G['sc'].to(torch.int32)}
    cs,ct,cd=solve(x.reshape(-1,3),y.reshape(-1,3),compressed,.1,.3,100,2)
    for a,b in ((cs,src),(ct,dst),(cd,diag)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_public_batch_matches_each_scene_and_does_not_mix():
    torch.manual_seed(8);x=torch.randn(2,67,3,device='cuda')*.3;y=torch.randn_like(x)*.3
    config=dict(k=32,epsilons=(.1,.02),penalty=.3,iterations=50)
    with torch.cuda.amp.autocast():z,info=public_match(x,y,**config)
    assert z.shape==x.shape and z.dtype==torch.float32 and info['parallel_batch'] and not info['hard_assignment']
    for b in range(2):
        ref,_=public_match(x[b],y[b],**config)
        torch.testing.assert_close(z[b],ref,atol=2e-5,rtol=3e-4)
    other=y.clone();other[1]+=100
    changed,_=match(x,other,**config)
    torch.testing.assert_close(changed[0],z[0],rtol=3e-5,atol=2e-6)
    identical,_=match(x,x,k=32,epsilons=(.1,),penalty=.3,iterations=150)
    torch.testing.assert_close(identical,x,rtol=3e-5,atol=2e-6)
    with pytest.raises(ValueError):match(x.half(),y.half())


def test_default_package_entry_selects_latest_match():
    from fpsgen.vggi.point_transport import match as recommended
    torch.manual_seed(17);x=torch.randn(1,67,3,device='cuda');y=torch.randn_like(x)
    out,info=recommended(x,y,epsilons=(.1,),iterations=3)
    ref,_=match(x,y,epsilons=(.1,),iterations=3,graph_backend='linear',reuse_knn_bounds=True)
    torch.testing.assert_close(out,ref,atol=3e-6,rtol=3e-5)
    assert info['implementation']=='exact_knn_bounds_linear_union_fp32_v1'
    assert info['reuse_knn_bounds'] and not info['hard_assignment']
