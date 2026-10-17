"""CPU contracts, genuine small U-Net gradients and state restoration (not CUDA)."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet,OccupancyGridSpec
from fpsgen.hier_occ_flow.stage1_training import validate_config,verify_stage1_resume,state_sha
from fpsgen.hier_occ_flow.stage1_monitoring import gradient_stats,sample_parameters,sample_changes
from fpsgen.hier_occ_flow.trainer import atomic_checkpoint,update_ema,rng_state,restore_rng
from fpsgen.hier_occ_flow.stage1_occupancy import masked_occ_loss


def config():
    return {'geometry':{'parent_resolution':[384,384,32],'child_split':[2,2,2]},
        'model':{'objective':'fm','initialization':'scratch','pretrained_encoder':None,'input_feature_version':'coarse_f64_v1'},
        'loss':{'fm':1.},'flow':{'source':'gaussian','sigma':1.,'target':'binary_symmetric'},
        'training':{'physical_gpus':[2,3],'batch_per_gpu':1,'precision':'fp16','successful_updates':100000}}


def test_stage1_config_locks384_occupancy_and_scratch():
    cfg=config();assert validate_config(cfg).fine_resolution==(768,768,64)
    for changed in ({'pretrained_encoder':'old.pt'},{'initialization':'pretrained'}):
        bad=copy.deepcopy(cfg);bad['model'].update(changed)
        with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['loss']['occupancy']=.1
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['geometry']['maximum']=[60.,50.,4.4]
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['geometry']['parent_resolution']=[256,256,32]
    with pytest.raises(ValueError):validate_config(bad)


@pytest.mark.parametrize('objective',['fm','bce'])
@pytest.mark.parametrize('batch',[1,2])
def test_real_unet_unit_grid_gradients_and_no_unused_parameters(objective,batch):
    #8-cubed tests layer math only; it is never a384-scene smoke result.
    torch.manual_seed(2);m=OccupancyUNet(objective,activation_checkpointing=True)
    state=torch.randn(batch,1,8,8,8);observed=torch.rand_like(state)>.8
    condition=[torch.randn(batch,128,n,n,n,requires_grad=True) for n in (4,2,1)]
    result=m(state if objective=='fm' else torch.zeros_like(state),observed,condition,torch.full((batch,),.4) if objective=='fm' else None)
    loss=masked_occ_loss(result,torch.randn_like(result) if objective=='fm' else observed.float(),torch.ones_like(observed),objective=objective)
    loss.backward()
    assert result.shape==state.shape and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(c.grad is not None and c.grad.abs().sum()>0 for c in condition)
    assert gradient_stats(m)['nonzero_parameter_ratio']>0
    assert gradient_stats(m.output)['l2']>0
    if objective=='bce':
        with pytest.raises(ValueError):m(state,observed,condition,torch.tensor(.5))


def test_initialization_repeatability_and_parameter_updates():
    torch.manual_seed(5);a=OccupancyUNet('fm');torch.manual_seed(5);b=OccupancyUNet('fm')
    assert state_sha(a)==state_sha(b)
    torch.manual_seed(6);c=OccupancyUNet('fm');assert state_sha(a)!=state_sha(c)
    sample=sample_parameters(a)
    with torch.no_grad():next(a.parameters()).add_(.01)
    _,changes=sample_changes(a,sample);assert changes['sample_delta_l2']>0
    assert changes['sample_changed_elements']>0


def test_new_roi_agrees_with_open_sphere_and_height():
    spec=OccupancyGridSpec(nx=384,ny=384,nz=32)
    xyz=torch.tensor([[0.,0.,-4.],[0.,0.,4.4],[50.,0.,0.],[0.,0.,0.],[49.9,0.,4.]])
    assert spec.valid_points(xyz).tolist()==[False,False,False,True,False]


def test_tokenizer_coarse_indices_use_configured_precision():
    from fpsgen.hier_occ_flow.stage1_network import occupancy_lidar_features
    spec=OccupancyGridSpec(nx=384,ny=384,nz=32)
    xyz=torch.tensor([[.5208333,0.,0.],[.5208334,0.,0.],[0.,.5208333,-3.9]],dtype=torch.float32)
    features,indices=occupancy_lidar_features(xyz,spec)
    expected=torch.floor((xyz.double()-xyz.new_tensor(spec.origin,dtype=torch.float64))/(xyz.new_tensor(spec.size,dtype=torch.float64)*2)).int()
    assert torch.equal(indices[0][0],expected)
    assert features.shape==(1,1,3,6)


def test_cpu_state_roundtrip_rejects_historical_resume(tmp_path):
    torch.manual_seed(8);model=torch.nn.Linear(2,2);ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters())
    scheduler=torch.optim.lr_scheduler.StepLR(opt,1);scaler=torch.cuda.amp.GradScaler(enabled=False)
    gen=torch.Generator().manual_seed(9);identity={'experiment':'hs-occflow-stage1-occ384-v1','geometry_sha256':'384'}
    def step():
        opt.zero_grad();x=torch.randn(3,2,generator=gen);loss=model(x).square().mean();loss.backward();opt.step();scheduler.step();update_ema(ema,model,.9)
        return float(loss)
    for i in range(20):step()
    snapshot={'identity':identity,'initialization_mode':'scratch','model':copy.deepcopy(model.state_dict()),'ema':copy.deepcopy(ema.state_dict()),
        'optimizer':copy.deepcopy(opt.state_dict()),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(gen,torch.device('cpu'))}
    atomic_checkpoint(tmp_path/'state.pt',snapshot);expected=step();expected_sha=state_sha(model);ema_sha=state_sha(ema)
    value=verify_stage1_resume(torch.load(tmp_path/'state.pt'),identity)
    model.load_state_dict(value['model']);ema.load_state_dict(value['ema']);opt.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler']);scaler.load_state_dict(value['scaler']);restore_rng(value['rng'],gen,torch.device('cpu'))
    assert step()==expected and state_sha(model)==expected_sha and state_sha(ema)==ema_sha
    with pytest.raises(ValueError):verify_stage1_resume(value,{'experiment':'old_density'})
    unsafe=dict(value);unsafe['parameter_sync_pass']=False
    with pytest.raises(ValueError):verify_stage1_resume(unsafe,identity)


def test_stage1_input_only_export_rejects_gt():
    root=Path(__file__).resolve().parents[1]/'scripts/research_v2/hier_occ_flow'
    import sys
    sys.path.insert(0,str(root))
    from export_stage1_occ import validate_inputs
    validate_inputs([{'id':'00_000000','input':'input.npy'}])
    with pytest.raises(ValueError):validate_inputs([{'id':'00','input':'x','gt':'oracle.ply'}])


def test_parent_or_matches_direct_raw_voxelization():
    from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
    from fpsgen.hier_occ_flow.voxelization import voxelize_surface
    g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    points=np.random.default_rng(3).uniform([-49,-49,-3.9],[49,49,4.3],size=(10000,3))
    ids,masks,_=voxelize_surface(points,g)
    fine,valid=g.voxelize_points(points)
    direct=np.floor((points[valid]-np.array(g.origin))/np.array(g.parent_size)).astype(np.int64)
    assert np.array_equal(ids,np.unique(g.flatten(direct))) and np.all(masks!=0)


@pytest.mark.parametrize('nfe',[1,2,4,8])
def test_stage1_actual_sampling_method_constant_velocity(nfe):
    from types import SimpleNamespace
    from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
    class Constant(torch.nn.Module):
        def forward(self,state,observed,condition,time=None):return torch.full_like(state,.3)
    model=Stage1DirectOccupancy.__new__(Stage1DirectOccupancy);torch.nn.Module.__init__(model)
    model.objective='fm';model.geometry=SimpleNamespace(field_shape=(2,3,4),observed_mask=lambda xyz:torch.zeros(1,1,2,3,4,dtype=torch.bool))
    model.flow=Constant();model.register_buffer('valid',torch.ones(1,1,2,3,4,dtype=torch.bool))
    source=torch.randn((1,1,2,3,4),generator=torch.Generator().manual_seed(4))
    score,state=model.sample(torch.zeros(2,3),generator=torch.Generator().manual_seed(4),nfe=nfe,condition=())
    torch.testing.assert_close(state,source+.3)
    torch.testing.assert_close(score,((source+.3+1)*.5).clamp(0,1))


def test_parallel_shards_atomic_assembly_without_data_copy(tmp_path):
    import sys
    root=Path(__file__).resolve().parents[1]/'scripts/research_v2/hier_occ_flow';sys.path.insert(0,str(root))
    from prepare_occ384_parallel import assemble
    from fpsgen.hier_occ_flow.storage import SparseShardWriter,SparseShardDataset
    from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
    g=SparseHierarchyGeometry(parent_resolution=(16,16,8),child_split=(2,2,2))
    for seq,frame in [('00',0),('05',2)]:
        writer=SparseShardWriter(tmp_path/'parts'/seq/'train',g,'source')
        writer.add(seq,frame,np.array([1,4],dtype=np.uint32),np.array([1,3],dtype=np.uint8));writer.flush()
    assemble(tmp_path,'train',g,'source',2)
    ds=SparseShardDataset(tmp_path/'train',g,verify=True)
    assert len(ds)==2 and [ds[i]['sequence'] for i in range(2)]==['00','05']
    assert (tmp_path/'train/shard_0000.ids.bin').stat().st_ino==(tmp_path/'parts/00/train/shard_0000.ids.bin').stat().st_ino
    assemble(tmp_path,'train',g,'source',2)


def test_parent_metrics_dense_membership_keeps_missing_false_negatives():
    from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics
    from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
    g=SparseHierarchyGeometry(parent_resolution=(4,4,2),child_split=(2,2,2))
    score=np.zeros((1,1,2,4,4),dtype=np.float32);score.ravel()[[1,3]]=1
    record={'parent_ids':np.array([1,2],dtype=np.uint32),'child_occ_mask':np.array([1,7],dtype=np.uint8)}
    result=parent_metrics(score,record,g,np.ones_like(score,dtype=bool),.5)
    assert result['tp']==1 and result['fp']==1 and result['fn']==1
    assert result['iou']==pytest.approx(1/3) and result['fine_recall_ceiling']==pytest.approx(.25)


def test_viewpoint_lookup_exact_membership_vs_original():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_semantickitti import viewpoint_mask
    from fpsgen.hier_occ_flow.viewpoint_lookup import viewpoint_mask_lookup
    rng=np.random.default_rng(4)
    for scan in (rng.uniform(-50,50,(100,3)),np.zeros((0,3))):
        points=np.concatenate((rng.uniform(-50,50,(2000,3)),np.array([[0,10,-10],[50,-50,4.4]])))
        assert np.array_equal(viewpoint_mask(scan,points,10.),viewpoint_mask_lookup(scan,points,10.))


def test_formal_scope_cannot_disguise_short_smoke(tmp_path):
    from fpsgen.hier_occ_flow.stage1_training import run_stage1
    cfg=config();cfg['scope']='formal'
    with pytest.raises(ValueError,match='full100K budget'):run_stage1(cfg,tmp_path,max_updates=4)


def test_pilot_cannot_satisfy_formal30min_monitor(monkeypatch,tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research_v2/hier_occ_flow'))
    import monitor_stage1_occ as monitor
    monkeypatch.setattr(monitor,'snapshot',lambda output:{'scope':'pilot'})
    monkeypatch.setattr(sys,'argv',['monitor','--output',str(tmp_path),'--report',str(tmp_path/'report.json'),'--observe'])
    with pytest.raises(ValueError,match='cannot satisfy formal30min'):monitor.main()


def test_batch_transition_preserves_global_cursor_and_epoch_order():
    from fpsgen.hier_occ_flow.stage1_batch_transition import batch_indices
    length=7;seed=23
    def expected(cursor,count):
        return [int(np.random.default_rng(seed+p//length).permutation(length)[p%length]) for p in range(cursor,cursor+count)]
    old=batch_indices(4,1,0,2,length,seed)+batch_indices(4,1,1,2,length,seed)
    assert old==expected(4,2)
    new=batch_indices(6,2,0,2,length,seed)+batch_indices(6,2,1,2,length,seed)
    assert new==expected(6,4)  # no skip at old→new batch/epoch boundaries


def test_batch_transition_rejects_unrelated_config_or_checkpoint(tmp_path):
    from fpsgen.hier_occ_flow.stage1_batch_transition import verify_batch_transition
    from fpsgen.hier_occ_flow.storage import file_sha
    import hashlib
    old=config();old['engineering_gate']='old_gate';old['training']['lr_encoder']=1e-4
    source=tmp_path/'config.json';source.write_text(json.dumps(old));ckpt=tmp_path/'resume.pt';ckpt.write_bytes(b'test')
    identity={'experiment':'hs-occflow-stage1-occ384-v1','objective':'fm','geometry_sha256':'384',
        'train_manifest_sha256':'train','development_manifest_sha256':'dev','world_size':2,'output':str(tmp_path),
        'config_sha256':hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()}
    value={'identity':identity,'initialization_mode':'scratch','parameter_sync_pass':True}
    new=copy.deepcopy(old);new['training']['batch_per_gpu']=2;new['engineering_gate']='new_gate'
    new['batch_transition']={'checkpoint':str(ckpt),'checkpoint_sha256':file_sha(ckpt),'source_config':str(source)}
    assert verify_batch_transition(value,identity,new,ckpt) is value
    bad=copy.deepcopy(new);bad['training']['lr_encoder']=2e-4
    with pytest.raises(ValueError,match='only batch'):verify_batch_transition(value,identity,bad,ckpt)
    bad=copy.deepcopy(new);bad['batch_transition']['checkpoint_sha256']='bad'
    with pytest.raises(ValueError,match='SHA mismatch'):verify_batch_transition(value,identity,bad,ckpt)
