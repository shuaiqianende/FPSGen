"""CPU contracts and actual small local-network math; never a PTv3 GPU claim."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.voxelization import pack_children,unpack_children,lookup_masks
from fpsgen.hier_occ_flow.condition import observed_children,sample_pyramid
from fpsgen.hier_occ_flow.stage2_candidates import parent_valid_field,oracle_parent_candidates,select_occupancy_candidates
from fpsgen.hier_occ_flow.stage2_condition import input_conditions,scene_seed
from fpsgen.hier_occ_flow.stage2_dataset import ConditionCache
from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.flow_matching import training_pair,fm_loss,bce_loss
from fpsgen.hier_occ_flow.inference import sample,export_occupancy
from fpsgen.hier_occ_flow.evaluation import fine_metrics
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,sampler_index,require_training_gate
from fpsgen.hier_occ_flow.stage2_trainer import own_resume
from fpsgen.hier_occ_flow.trainer import atomic_checkpoint,rng_state,restore_rng,update_ema
from fpsgen.hier_occ_flow.stage1_training import state_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import validate_stage1_identity,FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha,DATA_VERSION


@pytest.fixture
def g():return SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))


def center_ids(g,count=32):
    return g.flatten(np.stack((np.arange(count)+150,np.full(count,190),np.full(count,16)),1))


def test_384_768_packed_roundtrips(g):
    rng=np.random.default_rng(4);fine=rng.integers(0,g.fine_resolution,size=(1000,3))
    parent,child=g.split_fine(fine)
    assert g.k==8 and g.fine_resolution==(768,768,64)
    assert np.array_equal(g.compose(parent,child),fine)
    assert np.array_equal(g.unflatten(g.flatten(fine,fine=True),fine=True),fine)
    masks=np.arange(256,dtype=np.uint8)
    assert np.array_equal(pack_children(unpack_children(masks,g),g),masks)


def test_sphere_intersecting_mask_matches_child_union(g):
    ids=np.random.default_rng(3).integers(0,g.num_parents,5000)
    assert np.array_equal(parent_valid_field(g).ravel()[ids],g.valid_children(ids).any(1))


def test_oracle_nearby_negatives_neutral_scores_and_repeatability(g):
    ids=center_ids(g,100)
    a,s,source=oracle_parent_candidates(ids,g,seed=10)
    b,t,_=oracle_parent_candidates(ids,g,seed=10)
    assert len(a)==125 and source['nearby_negatives']==25
    assert np.isin(ids,a).all() and np.array_equal(a,b) and np.all(s==1) and np.array_equal(s,t)
    neg=np.setdiff1d(a,ids);coord=g.unflatten(neg);gt=g.unflatten(ids)
    assert (np.abs(coord[:,None]-gt[None]).max(2).min(1)<=1).all()
    assert np.all(lookup_masks(ids,np.ones(len(ids),dtype=np.uint8),neg)==0)


def test_predicted_ids_scores_zyx_threshold_topk_and_budget(g):
    ids=center_ids(g,3);scores=np.zeros(g.parent_resolution[::-1],np.float32);scores.ravel()[ids]=[.5,.8,.9]
    a,s,source=select_occupancy_candidates(scores,g,threshold=.5)
    assert np.array_equal(a,ids[1:]) and np.allclose(s,[.8,.9])
    with pytest.raises(ValueError):select_occupancy_candidates(scores.transpose(2,1,0),g)
    with pytest.raises(RuntimeError,match='no truncation'):select_occupancy_candidates(scores,g,budget=1)
    a,s,source=select_occupancy_candidates(scores,g,budget=1,overflow='topk_score')
    assert a.tolist()==[int(ids[2])] and source['budget_dropped']==1
    scores.ravel()[ids]=.9;scores[0,0,0]=1.
    a,_,_=select_occupancy_candidates(scores,g,strategy='topk',top_k=2)
    assert np.array_equal(a,ids[:2])


def test_world_zyx_multiscale_sampling_analytic(g):
    points=torch.tensor([[0.,0.,0.],[10.,-10.,1.]])
    pyramid=[]
    for shape in ((4,8,8),(2,4,4),(2,4,4)):
        z=(torch.arange(shape[0])+.5)*8.4/shape[0]-4
        y=(torch.arange(shape[1])+.5)*100/shape[1]-50
        x=(torch.arange(shape[2])+.5)*100/shape[2]-50
        field=x[None,None,:]+2*y[None,:,None]+3*z[:,None,None]
        pyramid.append(field[None,None].expand(1,128,*shape).contiguous())
    sampled=sample_pyramid(pyramid,points,g.origin,g.maximum)
    expected=points[:,0]+2*points[:,1]+3*points[:,2]
    assert sampled.shape==(2,384)
    torch.testing.assert_close(sampled,expected[:,None].expand(-1,384),atol=2e-5,rtol=1e-5)


class FakeAdapter:
    provenance={'type':'TEST_ONLY_CPU_STUB','checkpoint_sha256':'fake'}
    def __init__(self,g,ids):self.g=g;self.ids=ids;self.calls=0
    def __call__(self,xyz,**kwargs):
        self.calls+=1
        features=[torch.full((1,128,2,2,2),float(i+1)) for i in range(3)]
        score=torch.zeros(self.g.parent_resolution[::-1]);score.view(-1)[torch.as_tensor(self.ids)]=.8
        return features,score if kwargs['with_scores'] else None


def test_input_condition_has_no_gt_membership_and_observed_only_input(g):
    ids=center_ids(g,2);xyz=torch.tensor(g.centers(g.unflatten(ids)),dtype=torch.float32)
    adapter=FakeAdapter(g,ids)
    policy={'mode':'predicted','stage1_nfe':4,'selection':{'threshold':.5}}
    parent,c,source=input_conditions(xyz,g,adapter,policy,seed=1)
    assert parent.tolist()==ids.tolist() and c['lidar_features'].shape==(2,384) and adapter.calls==1
    assert c['lidar_features'].dtype==torch.float16
    assert c['observed_child'].sum()==2 and not any(t.requires_grad for t in c.values())
    assert set(c)=={'lidar_features','parent_score','observed_child','valid_child_mask'}
    with pytest.raises(ValueError,match='must not receive GT'):input_conditions(xyz,g,adapter,policy,seed=1,oracle_ids=ids)
    oracle={**policy,'mode':'oracle','oracle_score':'stage1'}
    _,c,_=input_conditions(xyz,g,adapter,oracle,seed=1,oracle_ids=ids)
    assert c['parent_score'].tolist()==pytest.approx([.8,.8])


def test_predicted_conditions_unchanged_when_gt_labels_change(g,tmp_path):
    from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
    ids=center_ids(g,2);directory=tmp_path/'00/input_';directory.mkdir(parents=True)
    np.save(directory/'000000.npy',g.centers(g.unflatten(ids)).astype(np.float32))
    cfg={'data_root':str(tmp_path),'candidates':{'mode':'predicted','stage1_nfe':4,'seed':1,'selection':{'threshold':.5}},
         'training':{'max_parent_tokens':100}}
    a={'sequence':'00','frame':0,'parent_ids':ids[:1].astype(np.uint32),'child_occ_mask':np.array([1],np.uint8)}
    b={**a,'parent_ids':ids[1:].astype(np.uint32),'child_occ_mask':np.array([2],np.uint8)}
    first=assemble_stage2_scene(a,cfg,g,torch.device('cpu'),FakeAdapter(g,ids))
    second=assemble_stage2_scene(b,cfg,g,torch.device('cpu'),FakeAdapter(g,ids))
    assert torch.equal(first[0],second[0])
    assert all(torch.equal(first[1][k],second[1][k]) for k in first[1])
    assert not torch.equal(first[2],second[2])


def test_observed_mask_ignores_semantic_column(g,tmp_path):
    from fpsgen.hier_occ_flow.dataset import input_xyz
    ids=center_ids(g,2);xyz=g.centers(g.unflatten(ids)).astype(np.float32)
    path=tmp_path/'input.npy';np.save(path,np.column_stack((xyz,[1,99])))
    a=observed_children(input_xyz(path),ids,g)
    np.save(path,np.column_stack((xyz,[12345,-1])))
    assert np.array_equal(a,observed_children(input_xyz(path),ids,g))


@pytest.mark.parametrize('objective',['fm','bce'])
def test_actual_local_math_all_eight_outputs_and_gradients(g,objective):
    torch.manual_seed(1);model=HierarchicalOccupancyModel(g,objective=objective,backbone='local')
    ids=torch.from_numpy(center_ids(g));valid=torch.from_numpy(g.valid_children(ids.numpy()))
    condition={'parent_score':torch.ones(len(ids)),'lidar_features':torch.randn(len(ids),384),
        'observed_child':torch.rand(len(ids),8)>.8,'valid_child_mask':valid}
    target=torch.rand(len(ids),8)>.7;generator=torch.Generator().manual_seed(5)
    if objective=='fm':
        pair=training_pair(target,valid,generator,t=torch.tensor(.3))
        torch.testing.assert_close(pair['state'],.7*pair['source']+.3*pair['target'])
        torch.testing.assert_close(pair['velocity'],pair['target']-pair['source'])
        result=model(ids,state=pair['state'],time=pair['time'],**condition);loss=fm_loss(result,pair['velocity'],valid)
    else:result=model(ids,**condition);loss=bce_loss(result,target,valid)
    loss.backward()
    assert result.shape==(len(ids),8) and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters())>0


@pytest.mark.parametrize('nfe',[1,2,4,8])
def test_euler_current_state_updated_and_same_initial_noise(g,nfe):
    class Constant(nn.Module):
        objective='fm'
        def __init__(self):super().__init__();self.times=[]
        def forward(self,ids,*,state,time,**condition):self.times.append(float(time));return torch.ones_like(state)*.25
    model=Constant().eval();ids=torch.tensor(center_ids(g,4));valid=torch.ones(4,8,dtype=torch.bool)
    endpoint,_,calls=sample(model,ids,{'valid_child_mask':valid},nfe=nfe,generator=torch.Generator().manual_seed(10))
    source=torch.randn(4,8,generator=torch.Generator().manual_seed(10))
    torch.testing.assert_close(endpoint,source+.25,atol=5e-7,rtol=1e-6)
    assert calls==nfe and model.times==[i/nfe for i in range(nfe)]


def test_stage1_missed_parent_fine_voxels_are_full_fns(g):
    ids=center_ids(g,2);masks=np.array([3,3],np.uint8)
    predicted=np.array([[1,1,0,0,0,0,0,0]],bool)
    result=fine_metrics(ids[:1],predicted,ids,masks,g,structural=True)
    assert result['fine']['tp']==2 and result['fine']['fn']==2 and result['fine']['recall']==.5
    assert result['missed_parent_fine_false_negatives']==2
    assert result['candidate']['fine_occupancy_recall_ceiling']==.5
    assert 'parent_boundary_contact_edges' in result and 'distance_bins' in result


def test_parent_wrapper_absolute_coords_and_inverse_order(g):
    from fpsgen.hier_occ_flow.ptv3_backbone import ParentPTv3
    class Identity(nn.Module):
        def forward(self,data):
            assert torch.equal(data['grid_coord'].min(0).values,torch.tensor([150,190,16],dtype=torch.int32))
            assert data['sparse_shape']==[385,385,33]
            return SimpleNamespace(feat=data['feat']*2)
    wrapper=ParentPTv3.__new__(ParentPTv3);nn.Module.__init__(wrapper);wrapper.net=Identity()
    ids=torch.tensor(center_ids(g,10));coords=g.unflatten(ids);features=torch.randn(10,64)
    perm=torch.tensor([9,2,6,7,0,1,4,5,3,8])
    result=wrapper(features[perm],coords[perm],g.centers(coords[perm]).float(),None,g.parent_resolution)
    torch.testing.assert_close(result,features[perm]*2)


def test_actual_ptv3_constructor_random_and_fixed_capacity_without_external_load(g):
    with patch('torch.load',side_effect=RuntimeError('No external Stage2 weights')):
        for objective in ('fm','bce'):
            model=HierarchicalOccupancyModel(g,objective=objective)
            assert all(p.requires_grad for p in model.parameters())
            assert not any(isinstance(m,nn.BatchNorm1d) for m in model.trunk.modules())
            assert model.decoder[-1].out_features==8
            assert sum(p.numel() for p in model.parameters())>100000


@pytest.mark.parametrize('objective',['fm','bce'])
def test_stage2_canonicalization_before_mlp_preserves_parent_output(g,objective):
    from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
    torch.manual_seed(3);model=Stage2OccupancyModel(g,objective=objective,backbone='local').eval()
    ids=torch.tensor(center_ids(g,40));permutation=torch.randperm(len(ids));conditions={
        'lidar_features':torch.randn(len(ids),384),'parent_score':torch.rand(len(ids)),
        'observed_child':torch.rand(len(ids),8)>.8,'valid_child_mask':torch.ones(len(ids),8,dtype=torch.bool)}
    if objective=='fm':conditions.update(state=torch.randn(len(ids),8),time=torch.tensor(.4))
    expected=model(ids,**conditions)
    shuffled={k:v[permutation] if k!='time' else v for k,v in conditions.items()}
    actual=model(ids[permutation],**shuffled)[torch.argsort(permutation)]
    assert torch.equal(expected,actual)


def test_stage2_single_scene_norm_matches_instance_norm_without_running_buffers():
    from fpsgen.hier_occ_flow.stage2_model import single_scene_norm
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point,PointInstanceNorm
    norm=single_scene_norm(PointInstanceNorm)(8)
    features=torch.randn(100,8,requires_grad=True);point=Point(feat=features,batch=torch.zeros(100,dtype=torch.long))
    result=norm(point).feat
    expected=torch.nn.functional.instance_norm(features.T[None]).squeeze(0).T
    torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-5)
    result.square().mean().backward();assert torch.isfinite(features.grad).all()
    assert not list(norm.buffers())


def test_stage2_pooling_serialization_never_shuffles_in_train_or_eval(g):
    from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import SerializedPooling
    for objective in ('bce','fm'):
        model=Stage2OccupancyModel(g,objective=objective)
        for training in (True,False):
            model.train(training)
            pools=[m for m in model.modules() if isinstance(m,SerializedPooling)]
            assert pools and all(m.shuffle_orders is False for m in pools)


def test_cache_identity_content_and_budget(g,tmp_path):
    cache=ConditionCache(tmp_path,max_bytes=1024**2);ids=torch.tensor(center_ids(g,4))
    c={'parent_score':torch.ones(4),'lidar_features':torch.randn(4,384).half(),
        'observed_child':torch.zeros(4,8,dtype=torch.bool),'valid_child_mask':torch.ones(4,8,dtype=torch.bool)}
    identity={'checkpoint':'sha','seed':1,'mode':'predicted'}
    cache.write(identity,ids,c,{'gt_used':False},g)
    restored=cache.read(identity,torch.device('cpu'),g)
    assert torch.equal(restored[0],ids) and torch.equal(restored[1]['lidar_features'],c['lidar_features'])
    assert cache.read({**identity,'checkpoint':'other'},torch.device('cpu'),g) is None
    with pytest.raises(RuntimeError,match='budget'):ConditionCache(tmp_path/'small',max_bytes=1).write(identity,ids,c,{},g)
    next(tmp_path.glob('*.npz')).write_bytes(b'corrupt')
    with pytest.raises(ValueError):cache.read(identity,torch.device('cpu'),g)


def test_resume_actual_small_model_optimizer_ema_rng_and_sampler(g,tmp_path):
    torch.manual_seed(8);model=nn.Linear(8,8);ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters())
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1);gen=torch.Generator().manual_seed(4)
    def step():
        optimizer.zero_grad();loss=model(torch.randn(32,8,generator=gen)).square().mean()
        loss.backward();optimizer.step();scheduler.step();update_ema(ema,model,.9);return float(loss)
    for _ in range(20):step()
    identity={'experiment':'hs-occflow-stage2-384-v1','stage1_checkpoint_sha256':'new_occ384','geometry_sha256':g.sha256}
    state={'identity':identity,'initialization':'scratch_stage2','parameter_sync_pass':True,'model':copy.deepcopy(model.state_dict()),
        'ema':copy.deepcopy(ema.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':scheduler.state_dict(),
        'rng':rng_state(gen,torch.device('cpu')),'sampler':{'attempts':20}}
    atomic_checkpoint(tmp_path/'own.pt',state);expected=step();expected_hash=state_sha(model)
    value=own_resume(torch.load(tmp_path/'own.pt'),identity)
    model.load_state_dict(value['model']);ema.load_state_dict(value['ema']);optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler'])
    restore_rng(value['rng'],gen,torch.device('cpu'))
    assert step()==expected and state_sha(model)==expected_hash
    assert sampler_index(20,0,2,17870,20261018)==sampler_index(value['sampler']['attempts'],0,2,17870,20261018)
    with pytest.raises(ValueError):own_resume(value,{**identity,'stage1_checkpoint_sha256':'old_density'})


def test_all_actual_configs_and_no_formal_gate_bypass(g,tmp_path):
    paths=list(Path('configs/research_v2/hier_occ_flow').glob('stage2_384_*.yaml'))
    assert len(paths)>=8
    for path in paths:
        cfg,geometry=load_stage2_config(path);assert geometry==g
        # A real local engineering gate can turn green while experiments run.
        # The missing-evidence unit test must own its fixture, not that artifact.
        cfg['engineering_gate']=str(tmp_path/'missing_gate.json')
        with pytest.raises(ValueError,match='gate missing'):require_training_gate(cfg,g)
    assert scene_seed('00',1,3)==scene_seed('00',1,3)!=scene_seed('00',2,3)


def test_evidence_never_turns_missing_gpu_into_pass(g,tmp_path):
    from fpsgen.hier_occ_flow.stage2_evidence import build_engineering_gate
    gate=build_engineering_gate(g.sha256,'new384')
    assert gate['status']=='PENDING_REAL_GPU_EVIDENCE'
    assert not gate['full_scene_gpu'] and not gate['ddp_resume'] and not gate['matched_bce_fm_1k']
    path=tmp_path/'fake.json';atomic_json(path,{'status':'PASS_GPU_FULL_SCENE','geometry_sha256':g.sha256,
                                             'stage1_checkpoint_sha256':'wrong'})
    with pytest.raises(ValueError,match='identity mismatch'):build_engineering_gate(g.sha256,'new384',gpu_reports=[path])


def test_resume_evidence_requires_exact_rng_cursor_optimizer_step():
    from fpsgen.hier_occ_flow.stage2_evidence import compare_resume,compare_tree
    assert compare_tree({'step':torch.tensor(21.)},{'step':torch.tensor(21.000001)})
    assert not compare_tree(torch.tensor([1.]),torch.tensor([1.000001]))
    assert compare_tree(torch.tensor([1],dtype=torch.uint8),torch.tensor([2],dtype=torch.uint8))
    assert compare_resume({}, {})['status']=='FAIL_DDP_RESUME'
    value={'identity':{'experiment':'hs-occflow-stage2-384-v1','world_size':2},'initialization':'scratch_stage2',
           'successful_updates':21,'attempts':21,'amp_skips':0,'consecutive_failures':0,
           'model':{'weight':torch.ones(2)},'ema':{'weight':torch.ones(2)},'optimizer':{'step':torch.tensor(21.)},
           'scheduler':{'last_epoch':21},'scaler':{},'rank_rng':[{'torch':torch.tensor([1],dtype=torch.uint8)}]*2,
           'sampler':{'attempts':21},'parameter_sync_pass':True,'rank_hashes':[{'online':'a','ema':'a'}]*2}
    assert compare_resume(value,copy.deepcopy(value))['status']=='PASS_DDP_RESUME'
    other=copy.deepcopy(value);other['sampler']['attempts']=20
    assert compare_resume(value,other)['status']=='FAIL_DDP_RESUME'


def test_stage1_adapter_identity_cannot_differ_from_gate(g,tmp_path):
    cfg=__import__('yaml').safe_load(Path('configs/research_v2/hier_occ_flow/stage2_384_oracle_fm.yaml').read_text())
    cfg['geometry']=g.to_dict();cfg['stage1']['adapter']['checkpoint_sha256']='wrong'
    path=tmp_path/'bad.yaml';path.write_text(__import__('yaml').safe_dump(cfg))
    with pytest.raises(ValueError,match='identity must agree'):load_stage2_config(path)


def test_input_only_deployment_export_is_sparse_fine768(g,tmp_path):
    ids=center_ids(g,2);occ=np.zeros((2,8),bool);occ[:,3]=True
    export_occupancy(tmp_path/'out',ids,occ,g)
    with np.load(tmp_path/'out/fine_occupancy.npz') as data:
        assert data['child_occ_mask'].tolist()==[8,8]
    metadata=json.loads((tmp_path/'out/manifest.json').read_text())
    assert metadata['points']==2 and metadata['geometry']['parent_resolution']==[384,384,32]


def test_diagnostic_ply_contains_missed_parent_false_negative(g,tmp_path):
    from fpsgen.hier_occ_flow.stage2_visualization import export_evaluation_diagnostics
    ids=center_ids(g,2);occ=np.zeros((1,8),bool);occ[0,0]=True
    export_evaluation_diagnostics(tmp_path/'visual',np.zeros((2,3)),ids[:1],occ,ids,np.ones(2,np.uint8),g)
    manifest=json.loads((tmp_path/'visual/manifest.json').read_text())
    assert manifest['false_negatives']==1 and manifest['false_positives']==0
    assert b'element vertex 1\n' in (tmp_path/'visual/false_negative.ply').read_bytes()[:150]


def test_stage1_identity_rejects_density_and_sha_before_load(g,tmp_path):
    with pytest.raises(ValueError,match='binary Occupancy'):validate_stage1_identity({'stage':'stage1'},{},g)
    path=tmp_path/'bad.pt';path.write_bytes(b'not weights')
    with patch('torch.load',side_effect=RuntimeError('Must not be reached')):
        with pytest.raises(ValueError,match='SHA changed'):
            FrozenStage1OccupancyAdapter(path,tmp_path/'config.json',checkpoint_sha256='incorrect')
