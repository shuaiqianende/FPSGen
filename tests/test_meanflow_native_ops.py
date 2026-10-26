import pytest
import torch
from fpsgen.hier_occ_flow.meanflow_native_ops import (
    sequence_groups, attention_tangent, segment_max_jvp, native_jvp_mode,
    install_native_jvp_bridges, flash_attention_jvp)


def reference_attention(x, lengths, scale):
    parts=[]; start=0
    for length in lengths:
        q,k,v=x[start:start+length].permute(1,2,0,3).unbind(0)
        parts.append((((q@k.transpose(-1,-2))*scale).softmax(-1)@v).transpose(0,1))
        start+=length
    return torch.cat(parts)


def test_chunked_attention_analytic_tangent_matches_double_autograd_with_ragged_windows():
    torch.manual_seed(42)
    lengths=[3,7,7,2];cu=[0]
    for length in lengths:cu.append(cu[-1]+length)
    groups=sequence_groups(cu)
    assert groups==((0,3,3),(3,17,7),(17,19,2))
    x=torch.randn(sum(lengths),3,2,4,dtype=torch.double)
    dx=torch.randn_like(x);scale=.31
    _,expected=torch.func.jvp(lambda z:reference_attention(z,lengths,scale),(x,),(dx,))
    actual=attention_tangent(x,dx,groups,scale,chunk_sequences=1)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def test_segment_max_uses_native_tie_subgradient_and_empty_segment():
    import torch_scatter
    x=torch.tensor([[1.,2.],[1.,3.],[-2.,4.],[7.,4.]],requires_grad=True)
    dx=torch.tensor([[2.,3.],[5.,7.],[11.,13.],[17.,19.]])
    ptr=torch.tensor([0,2,2,4],dtype=torch.long)
    primal,tangent=torch.func.jvp(lambda z:segment_max_jvp(z,ptr),(x,),(dx,))
    native,arg=torch_scatter.segment_max_csr(x,ptr)
    assert torch.equal(primal,native)
    expected=torch.cat((dx,torch.zeros_like(dx[:1]))).gather(0,arg.clamp(max=len(x)))
    assert torch.equal(tangent,expected)
    cot=torch.randn_like(primal)
    grad,=torch.autograd.grad(native,x,cot)
    torch.testing.assert_close((tangent*cot).sum(),(grad*dx).sum())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA SpConv required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_subm_native_primal_bias_free_tangent_and_reverse_dot(dtype):
    import spconv.pytorch as spconv
    torch.manual_seed(17)
    coords=torch.cartesian_prod(torch.arange(3),torch.arange(3),torch.arange(2))
    coords=torch.cat((torch.zeros(len(coords),1,dtype=torch.long),coords),1).int().cuda()
    conv=spconv.SubMConv3d(8,8,3,padding=1,bias=True,indice_key='jvp_test').cuda().to(dtype)
    net=torch.nn.Sequential(conv);install_native_jvp_bridges(net)
    x=torch.randn(len(coords),8,device='cuda',dtype=dtype,requires_grad=True);dx=torch.randn_like(x)
    def fn(z):return conv(spconv.SparseConvTensor(z,coords,[5,5,5],1)).features
    native=fn(x)
    with torch.no_grad(),native_jvp_mode():
        primal,tangent=torch.func.jvp(fn,(x,),(dx,))
    assert torch.equal(native,primal)
    cot=torch.randn_like(native)
    grad,=torch.autograd.grad(native,x,cot)
    lhs=(tangent.float()*cot.float()).sum();rhs=(grad.float()*dx.float()).sum()
    denominator=tangent.float().norm()*cot.float().norm()+grad.float().norm()*dx.float().norm()
    assert float((lhs-rhs).abs()/denominator)<(8e-4 if dtype==torch.float16 else 2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA FlashAttention required')
def test_flash_native_primal_and_ragged_analytic_tangent_against_double_and_native_reverse():
    from flash_attn import flash_attn_varlen_qkvpacked_func
    torch.manual_seed(7);lengths=[3,64,17,128];cu=[0]
    for length in lengths:cu.append(cu[-1]+length)
    groups=sequence_groups(cu);cu=torch.tensor(cu,device='cuda',dtype=torch.int32)
    qkv=(torch.randn(sum(lengths),3,4,16,device='cuda')*.5).half().requires_grad_(True)
    tangent=(torch.randn_like(qkv)*.2);scale=.25
    native=flash_attn_varlen_qkvpacked_func(qkv,cu,128,dropout_p=0.,softmax_scale=scale)
    with torch.no_grad():
        primal,dy=torch.func.jvp(lambda z:flash_attention_jvp(z,cu,128,scale,groups),(qkv,),(tangent,))
    assert torch.equal(native,primal)
    _,reference=torch.func.jvp(lambda z:reference_attention(z,lengths,scale),
        (qkv.detach().double().cpu(),),(tangent.detach().double().cpu(),))
    torch.testing.assert_close(dy.double().cpu(),reference,atol=8e-5,rtol=2e-3)
    cot=torch.randn_like(native);grad,=torch.autograd.grad(native,qkv,cot)
    lhs=(dy.float()*cot.float()).sum();rhs=(grad.float()*tangent.float()).sum()
    denominator=dy.float().norm()*cot.float().norm()+grad.float().norm()*tangent.float().norm()
    assert float((lhs-rhs).abs()/denominator)<3e-4


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA AMP weight cache regression')
def test_detached_jvp_does_not_poison_later_trainable_amp_weight_casts():
    from fpsgen.hier_occ_flow.meanflow_jvp import endpoint_jvp
    layer=torch.nn.Linear(8,4).cuda()
    x=torch.randn(2,8,device='cuda');t=torch.tensor([.2,.4],device='cuda');s=t+.2
    def fn(z,a,b):return layer(z)*(1+a[:,None]+b[:,None])
    with torch.autocast('cuda',dtype=torch.float16):
        endpoint_jvp(fn,x,t,s,torch.randn_like(x),model=layer)
        prediction=fn(x,t,s)
        assert prediction.requires_grad
        prediction.square().mean().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().max()>0
