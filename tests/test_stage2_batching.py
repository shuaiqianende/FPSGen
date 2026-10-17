"""Packed scene math, equal scene loss, isolation and checkpoint layout."""
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel,single_scene_norm
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse


def scenes():
    values=[]
    for n in (13,29):
        ids=torch.arange(n)+1000000
        c={'lidar_features':torch.randn(n,384),'parent_score':torch.ones(n),'observed_child':torch.zeros(n,8,dtype=torch.bool),'valid_child_mask':torch.ones(n,8,dtype=torch.bool)}
        values.append((ids,c,torch.rand(n,8)>.5,{}))
    return values


def test_ragged_pack_no_padding_or_dropped_parents_and_times():
    ids,c,target,lengths=pack_scenes(scenes())
    assert lengths==[13,29] and ids.shape==(42,) and c['batch'].tolist()==[0]*13+[1]*29
    pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],torch.Generator().manual_seed(11))
    assert pair['time'].shape==(2,) and pair['time'][0]!=pair['time'][1]
    t=pair['time'][c['batch'],None]
    torch.testing.assert_close(pair['state'],(1-t)*pair['source']+t*pair['target'])
    torch.testing.assert_close(pair['velocity'],pair['target']-pair['source'])


def test_loss_weights_scenes_equally_not_tokens():
    pred=torch.cat([torch.ones(2,8),torch.ones(20,8)*3]);target=torch.zeros_like(pred)
    valid=torch.ones_like(pred,dtype=torch.bool);batch=torch.tensor([0]*2+[1]*20)
    assert scene_mean_mse(pred,target,valid,batch)==5


def test_packed_local_model_matches_separate_and_has_no_scene_leak():
    g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    torch.manual_seed(10);model=Stage2OccupancyModel(g,backbone='local').eval();items=scenes()
    ids,c,target,lengths=pack_scenes(items)
    pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],torch.Generator().manual_seed(3))
    output=model(ids,state=pair['state'],time=pair['time'],**c)
    offset=0;expected=[]
    for i,(local_ids,local_c,_,_) in enumerate(items):
        expected.append(model(local_ids,state=pair['state'][offset:offset+len(local_ids)],time=pair['time'][i],**local_c));offset+=len(local_ids)
    torch.testing.assert_close(output,torch.cat(expected),atol=1e-7,rtol=1e-5)
    modified=pair['state'].clone();modified[13:]+=10
    other=model(ids,state=modified,time=pair['time'],**c)
    assert torch.equal(output[:13],other[:13])
    loss=scene_mean_mse(output,pair['velocity'],c['valid_child_mask'],c['batch']);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_norm_is_per_scene_and_unchanged_parameter_layout():
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointInstanceNorm,Point
    norm=single_scene_norm(PointInstanceNorm)(4)
    a=torch.randn(15,4);b=torch.randn(27,4)*2+100
    value=norm(Point(feat=torch.cat((a,b)),batch=torch.tensor([0]*15+[1]*27))).feat
    expected=torch.cat([norm(Point(feat=x,batch=torch.zeros(len(x),dtype=torch.long))).feat for x in (a,b)])
    assert torch.equal(value,expected)
    assert set(norm.state_dict())=={'weight','bias'}
