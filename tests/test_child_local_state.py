import copy
import pytest
import torch
from fpsgen.hier_occ_flow.child_local_state import ChildLocalStateEncoder
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.state_condition_interaction import load_shared_baseline


G=SparseHierarchyGeometry(parent_resolution=(4,4,2),child_split=(2,2,2))


@pytest.mark.parametrize('mode',['position','position_attention'])
def test_xyz_child_order_and_zero_residual(mode):
    m=ChildLocalStateEncoder(G,mode=mode)
    assert torch.equal(m.relative_centers,torch.tensor([
        [-.5,-.5,-.5],[.5,-.5,-.5],[-.5,.5,-.5],[.5,.5,-.5],
        [-.5,-.5,.5],[.5,-.5,.5],[-.5,.5,.5],[.5,.5,.5]]))
    x=torch.randn(3,8);mask=torch.ones(3,8,dtype=torch.bool)
    assert torch.equal(m(x,mask),torch.zeros(3,64))


@pytest.mark.parametrize('mode',['position','position_attention'])
def test_gradients_open_after_first_update(mode):
    torch.manual_seed(9);m=ChildLocalStateEncoder(G,mode=mode)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.01)
    x=torch.randn(4,8,requires_grad=True);mask=torch.ones_like(x,dtype=torch.bool)
    target=torch.randn(4,64)
    (m(x,mask)-target).square().mean().backward()
    assert m.output.weight.grad.abs().sum()>0
    optimizer.step();optimizer.zero_grad()
    (m(x,mask)-target).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())


@pytest.mark.parametrize('mode',['position','position_attention'])
def test_mask_parent_independence_permutation_and_resume(mode):
    m=ChildLocalStateEncoder(G,mode=mode)
    torch.nn.init.normal_(m.output.weight,std=.1)
    x=torch.randn(3,8);mask=torch.ones(3,8,dtype=torch.bool)
    mask[1,2:]=False;mask[2]=False
    y=m(x,mask)
    changed=x.clone();changed[1,2:]=float('nan');changed[2]=float('nan')
    assert torch.equal(y,m(changed,mask)) and torch.equal(y[2],torch.zeros(64))
    changed=x.clone();changed[0]+=100
    assert torch.equal(y[1:],m(changed,mask)[1:])
    order=torch.tensor([2,0,1])
    assert torch.allclose(y[order],m(x[order],mask[order]),atol=1e-7,rtol=1e-6)
    restored=copy.deepcopy(m);restored.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(y,restored(x,mask))


@pytest.mark.parametrize('mode',['position','position_attention'])
def test_source_model_is_preserved_at_zero_initialization(mode):
    b0=HierarchicalOccupancyModel(G,backbone='local',width=16)
    candidate=HierarchicalOccupancyModel(G,backbone='local',width=16,local_state_encoding=mode)
    audit=load_shared_baseline(candidate,b0.state_dict())
    assert all(k.startswith('child_state_encoder.') for k in audit['new_keys'])
    ids=torch.tensor([1,5,2]);cond={'state':torch.randn(3,8),'time':torch.tensor(.2),
        'parent_score':torch.rand(3),'lidar_features':torch.randn(3,384),
        'observed_child':torch.zeros(3,8,dtype=torch.bool),'valid_child_mask':torch.ones(3,8,dtype=torch.bool)}
    assert torch.equal(b0(ids,**cond),candidate(ids,**cond))
    assert candidate.fuse[0].in_features==b0.fuse[0].in_features
    with torch.no_grad():candidate.child_state_encoder.output.weight.fill_(.1)
    with pytest.raises(ValueError,match='start at zero'):load_shared_baseline(candidate,b0.state_dict())


def test_wrong_geometry_and_mixed_research_variables_rejected():
    with pytest.raises(ValueError):ChildLocalStateEncoder(SparseHierarchyGeometry())
    with pytest.raises(ValueError):HierarchicalOccupancyModel(G,backbone='local',local_state_encoding='position',context_strategy='static')


def test_registered_configs_match_except_the_local_attention_switch():
    from pathlib import Path
    from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
    root=Path(__file__).resolve().parents[1]/'configs/research_v2/hier_occ_flow'
    configs=[load_stage2_config(root/f'stage2_384_child_local_{arm}_20k.yaml')[0]
             for arm in ('position','attention')]
    for value in configs:
        assert value['training']['seed']==20261032 and value['training']['batch_per_gpu']==2
        assert value['training']['successful_frame_exposures']==20000
        assert value['candidates']['stage1_nfe']==2
        value.pop('research_protocol');value['model'].pop('local_state_encoding')
    assert configs[0]==configs[1]
