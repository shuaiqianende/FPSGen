"""Real CUDA correctness/gradient/stream tests for independent scene Chamfer."""
import os
import pytest
import torch
from fpsgen.ops.chamfer_scene import SceneChamferDistance,load_scene_extension

pytestmark=pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES')!='1',reason='Explicit idleGPU1 tests only')


@pytest.fixture(scope='module',autouse=True)
def extension():return load_scene_extension()


@pytest.mark.parametrize('mode',['tiled','bvh','auto'])
@pytest.mark.parametrize('n,m',[(1,1),(7,35),(33,65),(257,519),(1001,2053)])
def test_squared_distances_and_original_indices(mode,n,m):
    gen=torch.Generator(device='cuda').manual_seed(n+m)
    p=torch.randn(n,3,device='cuda',generator=gen)
    q=torch.randn(m,3,device='cuda',generator=gen)
    actual=SceneChamferDistance(mode)(p,q)
    # Small dense oracle only, never used in scene benchmarks.
    pair=(p[:,None]-q[None]).square().sum(-1)
    dp,ip=pair.min(1);dq,iq=pair.min(0)
    torch.testing.assert_close(actual[0],dp,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(actual[1],dq,atol=1e-6,rtol=1e-6)
    assert torch.equal(actual[2],ip) and torch.equal(actual[3],iq)


@pytest.mark.parametrize('mode',['tiled','bvh'])
@pytest.mark.parametrize('zero_matches',[False,True])
def test_loss_and_both_coordinate_gradients(mode,zero_matches):
    gen=torch.Generator(device='cuda').manual_seed(391)
    p=torch.randn(91,3,device='cuda',generator=gen).requires_grad_()
    q=(p.detach().clone() if zero_matches else torch.randn(133,3,device='cuda',generator=gen)).requires_grad_()
    actual=SceneChamferDistance(mode).sqrt_loss(p,q)
    pair=(p.detach()[:,None]-q.detach()[None]).square().sum(-1)
    ip,iq=pair.argmin(1),pair.argmin(0)
    expected=.5*((p-q[ip]).norm(dim=-1).mean()+(q-p[iq]).norm(dim=-1).mean())
    ga=torch.autograd.grad(actual,(p,q),retain_graph=True)
    ge=torch.autograd.grad(expected,(p,q))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    for a,e in zip(ga,ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,e,atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('mode',['tiled','bvh'])
def test_duplicate_zero_and_exact_midpoint_ties(mode):
    p=torch.zeros(1031,3,device='cuda',requires_grad=True)
    q=torch.zeros(2071,3,device='cuda',requires_grad=True)
    metric=SceneChamferDistance(mode)
    dp,dq,ip,iq=metric(p,q)
    assert (dp==0).all() and (dq==0).all() and (ip==0).all() and (iq==0).all()
    loss=metric.sqrt_loss(p,q);gp,gq=torch.autograd.grad(loss,(p,q))
    assert (gp==0).all() and (gq==0).all()
    p=torch.tensor([[0.,0.,0.]],device='cuda');q=torch.tensor([[-1.,0.,0.],[1.,0.,0.]],device='cuda')
    assert metric(p,q)[2].item()==0


def test_ragged_and_dense_batch_keep_scene_separate():
    metric=SceneChamferDistance('bvh')
    p=torch.tensor([[0.,0.,0.]],device='cuda');q=torch.tensor([[3.,0.,0.],[5.,0.,0.]],device='cuda')
    rows=metric([p,q],[q,q]);assert rows[0][0].item()==9 and (rows[1][0]==0).all()
    torch.testing.assert_close(metric.sqrt_loss([p,q],[q,q]),torch.tensor(1.75,device='cuda'))
    dense=metric(torch.stack([p,p+10]),torch.stack([q,q+10]))
    assert dense[0].shape==(2,1) and torch.equal(dense[0][0],dense[0][1])


def test_current_cuda_stream_and_noncontiguous_inputs():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p=torch.randn(3,157,device='cuda').t().requires_grad_()
        q=torch.randn(3,203,device='cuda').t().requires_grad_()
        loss=SceneChamferDistance('bvh').sqrt_loss(p,q)
        gp,gq=torch.autograd.grad(loss,(p,q))
    stream.synchronize()
    assert torch.isfinite(gp).all() and torch.isfinite(gq).all()


def test_validation_and_auto_configuration():
    with pytest.raises(ValueError):SceneChamferDistance('fake')
    with pytest.raises(ValueError):SceneChamferDistance(leaf_size=0)
    with pytest.raises(ValueError):SceneChamferDistance()(torch.empty(0,3,device='cuda'),torch.zeros(1,3,device='cuda'))
    with pytest.raises(ValueError):SceneChamferDistance()(torch.zeros(1,3),torch.zeros(1,3))


@pytest.mark.parametrize('mode',['tiled','bvh'])
def test_fp32_squared_overflow_never_returns_invalid_indices(mode):
    p=torch.full((2,3),1e20,device='cuda');q=torch.zeros(3,3,device='cuda')
    dp,dq,ip,iq=SceneChamferDistance(mode)(p,q)
    assert torch.isinf(dp).all() and torch.isinf(dq).all()
    assert ((ip>=0)&(ip<len(q))).all() and ((iq>=0)&(iq<len(p))).all()
