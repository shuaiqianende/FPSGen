import copy
import torch
from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet
from fpsgen.hier_occ_flow.meanflow_stage1 import MeanFlowStage1
from fpsgen.hier_occ_flow.meanflow_stage2 import MeanFlowStage2
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.meanflow_jvp import endpoint_jvp


def stage1_case():
    base=OccupancyUNet('fm',widths=(8,8,8,8),activation_checkpointing=False)
    model=MeanFlowStage1(copy.deepcopy(base));state=torch.randn(2,1,8,8,8);observed=torch.zeros_like(state,dtype=torch.bool)
    condition=tuple(torch.randn(2,128,n,n,n) for n in (4,2,1));time=torch.tensor([.2,.4]);end=torch.tensor([.7,.8])
    return base,model,state,observed,condition,time,end


def test_real_stage1_conv_groupnorm_interpolation_cpu_jvp_and_zero_init():
    torch.set_num_threads(2);base,model,x,o,c,t,s=stage1_case()
    assert model.initialization_audit['shared_tensors_exact']
    assert torch.equal(base(x,o,c,t),model(x,o,c,t,time_end=s))
    fn=lambda z,a,b:model(z,o,c,a,time_end=b,validate_times=False)
    p,d=endpoint_jvp(fn,x,t,s,torch.randn_like(x),model=model)
    assert torch.isfinite(p).all() and torch.isfinite(d).all() and d.abs().sum()>0
    fn(x,t,s).square().mean().backward()
    assert model.base.embedding.interval.net[-1].weight.grad.abs().sum()>0


def test_stage1_cached_forward_receives_endpoint_and_original_precision_buffers():
    torch.set_num_threads(2);base,model,x,o,c,t,s=stage1_case();model.eval()
    with torch.no_grad():
        ctx=model.prepare_inference_context(o,c)
        assert torch.equal(model(x,o,c,t,time_end=s),model.forward_with_inference_context(x,ctx,t,time_end=s))
    frequencies=model.base.frequencies.clone();model.half()
    assert model.base.frequencies.dtype==torch.float32 and torch.equal(model.base.frequencies,frequencies)


def local_stage2():
    # CPU local surrogate is explicitly not a real PTv3/GPU JVP certificate.
    g=SparseHierarchyGeometry(parent_resolution=(2,2,2),child_split=(2,2,2))
    base=Stage2OccupancyModel(g,backbone='local',checkpointing=False)
    model=MeanFlowStage2(copy.deepcopy(base));ids=torch.tensor([0,3,6,1,2]);batch=torch.tensor([0,0,0,1,1])
    c=dict(batch=batch,lidar_features=torch.randn(5,384),parent_score=torch.rand(5),
        observed_child=torch.zeros(5,8,dtype=torch.bool),valid_child_mask=torch.ones(5,8,dtype=torch.bool))
    x=torch.randn(5,8);t=torch.tensor([.2,.4]);s=torch.tensor([.6,.9])
    return base,model,ids,c,x,t,s


def test_stage2_canonical_prepared_and_permutation_cpu_local_surrogate():
    base,model,ids,c,x,t,s=local_stage2()
    reference=base(ids,state=x,time=t,**c);actual=model(ids,state=x,time=t,time_end=s,**c)
    assert torch.equal(reference,actual)
    context=model.prepare_conditions(ids,**c)
    assert torch.equal(actual,model.forward_prepared(context,x,t,time_end=s))
    order=torch.tensor([4,2,0,3,1]);c2={k:v[order] for k,v in c.items()}
    shuffled=model(ids[order],state=x[order],time=t,time_end=s,**c2)
    assert torch.equal(actual,shuffled[torch.argsort(order)])
    for i in (0,1):
        selected=c['batch']==i;single={k:v[selected] for k,v in c.items() if k!='batch'}
        out=model(ids[selected],state=x[selected],time=t[i:i+1],time_end=s[i:i+1],**single)
        assert torch.allclose(actual[selected],out,atol=1e-7,rtol=1e-5)


def test_stage2_prepared_jvp_includes_state_and_time_and_lidar_projection_gradients():
    base,model,ids,c,x,t,s=local_stage2();context=model.prepare_conditions(ids,**c)
    fn=lambda z,a,b:model.forward_prepared(context,z,a,time_end=b)
    p,d=endpoint_jvp(fn,x,t,s,torch.ones_like(x),model=model)
    assert torch.isfinite(d).all() and d.abs().sum()>0
    fn(x,t,s).square().mean().backward()
    assert model.base.lidar[0].weight.grad.abs().sum()>0
    assert model.base.time.interval.net[-1].weight.grad.abs().sum()>0


def test_static_scene_norm_layout_keeps_original_statistics_and_native_jvp():
    from fpsgen.hier_occ_flow.meanflow_stage2 import _norm_forward
    from types import SimpleNamespace
    class Point(dict):
        def __getattr__(self,key):return self[key]
        def __setattr__(self,key,value):self[key]=value
    m=SimpleNamespace(_mf_layout_mode='capture',_mf_lengths=None,eps=1e-5,affine=True,
        weight=torch.ones(4,dtype=torch.double),bias=torch.zeros(4,dtype=torch.double))
    batch=torch.tensor([0,0,0,1,1]);x=torch.randn(5,4,dtype=torch.double)
    def original(z):
        values=[]
        for scene in z.split((3,2)):
            center=scene-scene.mean(0,keepdim=True);values.append(center*torch.rsqrt(center.square().mean(0,keepdim=True)+m.eps))
        return torch.cat(values)*m.weight+m.bias
    actual=_norm_forward(m,Point(feat=x,batch=batch)).feat
    assert m._mf_lengths==(3,2) and torch.equal(actual,original(x))
    m._mf_layout_mode='static';fn=lambda z:_norm_forward(m,Point(feat=z,batch=batch)).feat
    tangent=torch.randn_like(x)
    p,d=torch.func.jvp(fn,(x,),(tangent,));p2,d2=torch.func.jvp(original,(x,),(tangent,))
    assert torch.equal(p,p2) and torch.equal(d,d2)


def test_real_pooling_wrapper_recomputes_projection_and_reduction_from_new_features_cpu():
    from types import MethodType
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import SerializedPooling,PointInstanceNorm
    from fpsgen.hier_occ_flow.stage2_model import single_scene_norm
    from fpsgen.hier_occ_flow.ptv3_inference_topology import PTv3InferenceTopology
    from fpsgen.hier_occ_flow.meanflow_stage2 import _pool_forward
    coords=torch.tensor([[0,0,0],[1,0,0],[0,0,0],[2,0,0]]);batch=torch.tensor([0,0,1,1])
    topology=PTv3InferenceTopology.build(coords,coords.float(),batch,[9,9,9],('z','z-trans'),3,scene_count=2,level='multilevel')
    pool=SerializedPooling(4,8,norm_layer=single_scene_norm(PointInstanceNorm),act_layer=torch.nn.ReLU,shuffle_orders=False)
    pool._mf_original_forward=pool.forward;pool._mf_pool_mode='capture';pool._mf_pool_layout=None;pool.forward=MethodType(_pool_forward,pool)
    def point(x):
        p=topology.make_point(x);p['_validated_batch_size']=2;return p
    x=torch.randn(4,4);reference=pool(point(x));pool._mf_pool_mode='static';cached=pool(point(x))
    assert torch.equal(reference.feat,cached.feat)
    for key in ('coord','grid_coord','serialized_code','serialized_order','serialized_inverse','pooling_inverse'):
        assert torch.equal(reference[key],cached[key])
    y=torch.randn(4,4,requires_grad=True);actual=pool(point(y));pool._mf_pool_mode='original';expected=pool(point(y))
    assert torch.equal(actual.feat,expected.feat)
    actual.feat.square().sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert pool.proj.weight.grad is not None and torch.isfinite(pool.proj.weight.grad).all()
