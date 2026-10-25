import copy
import torch
from torch import nn
from fpsgen.hier_occ_flow.meanflow_jvp import endpoint_jvp,finite_difference_diagnostic
from fpsgen.hier_occ_flow.meanflow_math import meanflow_loss,sample_endpoints,scene_squared_errors
from fpsgen.hier_occ_flow.meanflow_interval import IntervalTimeEmbedding
from fpsgen.hier_occ_flow.meanflow_sampling import sample_meanflow,stage1_score


def test_forward_time_positive_identity_for_state_dependent_exponential_flow():
    x=torch.tensor([[.7],[-.3]],dtype=torch.double);t=torch.tensor([.2,.3],dtype=torch.double);s=torch.tensor([.8,.9],dtype=torch.double)
    a=.7;v=a*x
    def average(z,current,end):
        d=(end-current)[:,None]
        return z*torch.expm1(a*d)/d
    u,derivative=endpoint_jvp(average,x,t,s,v)
    assert torch.allclose(u,v+(s-t)[:,None]*derivative,atol=1e-12,rtol=1e-12)
    assert not torch.allclose(u,v-(s-t)[:,None]*derivative,atol=1e-4)
    fd=finite_difference_diagnostic(average,x,t,s,v,1e-5)
    assert torch.allclose(derivative,fd,atol=1e-9,rtol=1e-8)


def test_endpoint_jvp_matches_duration_chain_rule():
    x=torch.randn(2,3,dtype=torch.double);t=torch.tensor([.2,.4],dtype=torch.double);s=torch.tensor([.7,.9],dtype=torch.double);v=torch.randn_like(x)
    def duration(z,current,d):return .3*z.square()+2*current[:,None]+4*d[:,None]
    endpoint=lambda z,current,end:duration(z,current,end-current)
    _,a=endpoint_jvp(endpoint,x,t,s,v)
    _,b=torch.func.jvp(duration,(x,t,s-t),(v,torch.ones_like(t),-torch.ones_like(t)))
    _,wrong=torch.func.jvp(duration,(x,t,s-t),(v,torch.ones_like(t),torch.zeros_like(t)))
    assert torch.allclose(a,b,atol=1e-12) and not torch.allclose(a,wrong)
    assert torch.allclose(a,.6*x*v-2,atol=1e-12)


def test_diagonal_is_old_fm_loss_and_gradient_and_target_has_no_graph():
    weight=nn.Parameter(torch.tensor(.4));x=torch.randn(2,1,2,2,2);v=torch.randn_like(x);t=torch.tensor([.2,.6]);mask=torch.ones_like(x,dtype=torch.bool)
    fn=lambda z,current,end:weight*z+.2*current.reshape(2,1,1,1,1)
    loss,details=meanflow_loss(fn,x,t,t,v,mask)
    expected=((fn(x,t,t)-v).square()*mask).sum()/mask.sum()
    grad=torch.autograd.grad(loss,weight,retain_graph=True)[0];reference=torch.autograd.grad(expected,weight)[0]
    assert torch.equal(loss,expected) and torch.equal(grad,reference)
    assert not details['target'].requires_grad and details['prediction'].requires_grad


def test_masked_invalid_channels_never_reach_velocity_or_state():
    x=torch.tensor([[1.,999.],[2.,999.]])
    valid=torch.tensor([[True,False],[True,False]]);v=torch.ones_like(x)*5;t=torch.tensor([.2,.3]);s=t+.2
    seen=[]
    def fn(z,current,end):seen.append(z.detach().clone());return 2*z+current[:,None]
    loss,d=meanflow_loss(fn,x,t,s,v,valid,skip_all_diagonal=False)
    assert all(torch.equal(z[:,1],torch.zeros(2)) for z in seen)
    assert torch.equal(d['target'][:,1],torch.zeros(2)) and torch.isfinite(loss)


def test_bernoulli_per_scene_probability_and_independent_rng():
    source=torch.Generator().manual_seed(20262032);interval=torch.Generator().manual_seed(20263032)
    before=source.get_state().clone();t=torch.full((2,),.3);count=0
    for _ in range(3000):
        end,diagonal=sample_endpoints(t,interval);count+=int(diagonal.sum())
        assert bool(((end>=t)&(end<=1)).all())
    assert .73<count/6000<.77 and torch.equal(before,source.get_state())


def test_packed_scene_equal_weight_not_token_weight_and_adaptive_scene_mean():
    p=torch.tensor([[1.],[3.],[3.],[3.]]);target=torch.zeros_like(p);batch=torch.tensor([0,1,1,1]);valid=torch.ones_like(p,dtype=torch.bool)
    q=scene_squared_errors(p,target,valid,batch)
    assert torch.equal(q,torch.tensor([1.,9.])) and q.mean()==5


def test_interval_last_projection_learns_before_hidden_and_boundary_is_exact():
    torch.manual_seed(5);original=nn.Sequential(nn.Linear(3,5),nn.SiLU(),nn.Linear(5,4)).double()
    source=copy.deepcopy(original);adapted=IntervalTimeEmbedding(original,4,7).double()
    x=torch.randn(2,3,dtype=torch.double);d=torch.tensor([.3,.6],dtype=torch.double)
    with adapted.condition(d):y=adapted(x)
    assert torch.equal(y,source(x))
    y.square().sum().backward()
    assert adapted.interval.net[-1].weight.grad.abs().sum()>0
    assert adapted.interval.net[0].weight.grad.abs().sum()==0
    with torch.no_grad():adapted.interval.net[-1].weight.add_(-.1*adapted.interval.net[-1].weight.grad)
    adapted.zero_grad()
    with adapted.condition(d):adapted(x).square().sum().backward()
    assert adapted.interval.net[0].weight.grad.abs().sum()>0
    with adapted.condition(torch.zeros_like(d)):assert torch.equal(adapted(x),source(x))
    assert isinstance(copy.deepcopy(adapted),IntervalTimeEmbedding)


def test_sampling_uses_each_actual_endpoint_not_always_one_and_fp32_state():
    calls=[];source=torch.ones(2,1,dtype=torch.float16)
    def model(x,t,s):calls.append((x.clone(),t.clone(),s.clone()));return 2*x
    result=sample_meanflow(model,source,torch.ones_like(source,dtype=torch.bool),nfe=2,scenes=2)
    assert torch.equal(calls[0][1],torch.zeros(2)) and torch.equal(calls[0][2],torch.full((2,),.5))
    assert torch.equal(calls[1][1],torch.full((2,),.5)) and torch.equal(calls[1][2],torch.ones(2))
    assert torch.equal(calls[1][0],2*source.float()) and result.dtype==torch.float32
    assert torch.equal(stage1_score(torch.tensor([-2.,0.,2.])),torch.tensor([0.,.5,1.]))


def test_missing_native_forward_ad_is_not_silently_zero_or_finite_difference():
    import pytest
    class Unsupported(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x.square()
        @staticmethod
        def backward(ctx,grad):return grad
    x=torch.ones(2,2);t=torch.tensor([.2,.3]);s=t+.1
    with pytest.raises(RuntimeError,match='setup_context'):
        meanflow_loss(lambda z,a,b:Unsupported.apply(z),x,t,s,x,torch.ones_like(x,dtype=torch.bool))


def test_jvp_temporarily_disables_checkpoint_not_training_mode():
    class Probe(nn.Module):
        def __init__(self):super().__init__();self.checkpointing=True;self.activation_checkpointing=True;self.observed=[]
        def forward(self,x,t,s):
            self.observed.append((self.training,self.checkpointing,self.activation_checkpointing,torch.is_grad_enabled(),torch.is_inference_mode_enabled()))
            return x.square()+t[:,None]+s[:,None]
    model=Probe().train();x=torch.ones(2,2);t=torch.tensor([.2,.3]);s=t+.2
    endpoint_jvp(model,x,t,s,x,model=model)
    assert model.observed==[(True,False,False,False,False)]
    assert model.training and model.checkpointing and model.activation_checkpointing
