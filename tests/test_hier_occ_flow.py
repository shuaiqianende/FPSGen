import copy
import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.flow_matching import training_pair,fm_loss,bce_loss
from fpsgen.hier_occ_flow.trainer import atomic_checkpoint,load_resume,rng_state,restore_rng,seed_all,update_ema


def small_problem(k18=True):
    g=SparseHierarchyGeometry(parent_resolution=(8,8,8),child_split=(3,3,2) if k18 else (2,2,2))
    ids=torch.arange(16)+200; valid=g.valid_children(ids)
    condition={'parent_score':torch.ones(len(ids)),'observed_child':torch.zeros((len(ids),g.k)), 'valid_child_mask':valid}
    occ=torch.rand((len(ids),g.k))>.6
    return g,ids,condition,occ


@pytest.mark.parametrize('k18',[False,True])
def test_model_gradients_permutation_and_time(k18):
    torch.set_num_threads(2); seed_all(21); g,ids,c,occ=small_problem(k18)
    model=HierarchicalOccupancyModel(g,backbone='local',width=32,condition_mode='observed')
    gen=torch.Generator().manual_seed(7); pair=training_pair(occ,c['valid_child_mask'],gen)
    pred=model(ids,state=pair['state'],time=pair['time'],**c)
    loss=fm_loss(pred,pair['velocity'],c['valid_child_mask']); loss.backward()
    assert pred.shape==(16,g.k) and loss.isfinite()
    assert model.local[0].weight.grad.norm()>0 and model.decoder[-1].weight.grad.norm()>0
    assert model.time[0].weight.grad.norm()>0
    perm=torch.randperm(len(ids)); cp={k:v[perm] for k,v in c.items()}
    model.eval()
    permuted=model(ids[perm],state=pair['state'][perm],time=pair['time'],**cp)
    assert torch.allclose(permuted,pred[perm],atol=1e-6)


def test_source_independent_of_target_and_loss_masks():
    g,ids,c,occ=small_problem()
    a=training_pair(occ,c['valid_child_mask'],torch.Generator().manual_seed(3),torch.tensor(.25))
    b=training_pair(~occ,c['valid_child_mask'],torch.Generator().manual_seed(3),torch.tensor(.25))
    assert torch.equal(a['source'],b['source'])
    assert torch.allclose(a['target']-a['source'],a['velocity'])
    assert torch.allclose(a['state'],.75*a['source']+.25*a['target'])
    assert fm_loss(a['velocity'],a['velocity'],c['valid_child_mask'])==0
    with pytest.raises(ValueError): training_pair(occ,c['valid_child_mask'],torch.Generator(),torch.tensor([0.,1.]))


def test_direct_bce_has_no_flow_time_input():
    g,ids,c,occ=small_problem(False)
    model=HierarchicalOccupancyModel(g,objective='bce',backbone='local',condition_mode='observed',width=32)
    logits=model(ids,**c); loss=bce_loss(logits,occ,c['valid_child_mask']); loss.backward()
    assert loss.isfinite() and not hasattr(model,'time') and not hasattr(model,'local')
    with pytest.raises(ValueError): model(ids,time=torch.tensor(0.),**c)


def test_atomic_resume_reproduces_next_update_and_ema(tmp_path):
    torch.set_num_threads(2); seed_all(41); g,ids,c,occ=small_problem(False)
    def build():
        m=HierarchicalOccupancyModel(g,backbone='local',width=16,condition_mode='observed')
        o=torch.optim.AdamW(m.parameters(),lr=1e-3); s=torch.optim.lr_scheduler.StepLR(o,10,.9)
        return m,o,s
    m,o,s=build(); ema=copy.deepcopy(m); gen=torch.Generator().manual_seed(8)
    assert all(torch.equal(a,b) for a,b in zip(ema.parameters(),m.parameters()))
    def step(m,o,s,gen):
        pair=training_pair(occ,c['valid_child_mask'],gen); o.zero_grad()
        loss=fm_loss(m(ids,state=pair['state'],time=pair['time'],**c),pair['velocity'],c['valid_child_mask'])
        loss.backward(); o.step(); s.step(); return loss.detach()
    for _ in range(20): step(m,o,s,gen); update_ema(ema,m,.9)
    identity={'experiment':'test','geometry':g.sha256}
    atomic_checkpoint(tmp_path/'saved.pt',{'identity':identity,'model':m.state_dict(),'optimizer':o.state_dict(),
        'scheduler':s.state_dict(),'ema':ema.state_dict(),'rng':rng_state(gen,torch.device('cpu')),'successful_updates':20})
    expected=step(m,o,s,gen); update_ema(ema,m,.9)
    m2,o2,s2=build(); value=load_resume(tmp_path/'saved.pt',identity)
    m2.load_state_dict(value['model']); o2.load_state_dict(value['optimizer']); s2.load_state_dict(value['scheduler'])
    ema2=copy.deepcopy(m2); ema2.load_state_dict(value['ema']); restore_rng(value['rng'],gen,torch.device('cpu'))
    actual=step(m2,o2,s2,gen); update_ema(ema2,m2,.9)
    assert torch.equal(expected,actual)
    assert all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),m2.state_dict().values()))
    assert all(torch.equal(a,b) for a,b in zip(ema.state_dict().values(),ema2.state_dict().values()))
    assert s.state_dict()==s2.state_dict() and o.state_dict()['param_groups']==o2.state_dict()['param_groups']
    for key in o.state_dict()['state']:
        for name,item in o.state_dict()['state'][key].items():
            assert torch.equal(item,o2.state_dict()['state'][key][name])
    with pytest.raises(ValueError): load_resume(tmp_path/'saved.pt',{'experiment':'other'})


def test_actual_cpu_runner_fresh_and_resume(tmp_path):
    from fpsgen.hier_occ_flow.storage import SparseShardWriter
    from fpsgen.hier_occ_flow.voxelization import voxelize_surface
    from fpsgen.hier_occ_flow.trainer import run_training
    g=SparseHierarchyGeometry(parent_resolution=(8,8,8),child_split=(2,2,2))
    xyz=np.array([[0,0,0],[1,1,1]],np.float32); ids,masks,_=voxelize_surface(xyz,g)
    inp=tmp_path/'data/00/input_'; inp.mkdir(parents=True)
    np.save(inp/'000000.npy',np.column_stack((xyz,[100,200])))
    w=SparseShardWriter(tmp_path/'labels',g,'toy-source',shard_frames=1); w.add('00',0,ids,masks)
    cfg={'geometry':g.to_dict(),'train_dataset':str(tmp_path/'labels'),'data_root':str(tmp_path/'data'),
        'model':{'objective':'fm','backbone':'local','width':16,'condition_mode':'observed'},
        'candidates':{'mode':'oracle','negative_fraction':.25},
        'training':{'successful_updates':2,'seed':1,'lr':.001,'weight_decay':.01,'grad_clip':1.,'precision':'fp32',
          'warmup_updates':1,'checkpoint_every':1,'max_parent_tokens':1000,'ema_decay':.9},'validation':{'every_updates':0}}
    run_training(cfg,tmp_path/'run',max_updates=1)
    resume=tmp_path/'run/checkpoints/resume_latest.pt'
    run_training(cfg,tmp_path/'run',resume=resume,max_updates=2)
    assert torch.load(resume,map_location='cpu')['successful_updates']==2


def test_real_ptv3_construction_bce_has_no_unused_time_parameters():
    from unittest.mock import patch
    g=SparseHierarchyGeometry()
    with patch('torch.load',side_effect=AssertionError('Unexpected model weight read')):
        bce=HierarchicalOccupancyModel(g,objective='bce',backbone='ptv3')
        fm=HierarchicalOccupancyModel(g,objective='fm',backbone='ptv3')
    assert not any('time_modulation' in name for name,_ in bce.named_parameters())
    assert any('time_modulation' in name for name,_ in fm.named_parameters())
    assert bce.trunk.net.num_stages==fm.trunk.net.num_stages==2


def test_parent_adapter_keeps_absolute_grid_and_inverse_order():
    from types import SimpleNamespace
    from fpsgen.hier_occ_flow.ptv3_backbone import ParentPTv3
    g=SparseHierarchyGeometry(); ids=torch.tensor([2002,2,1000]); coords=g.unflatten(ids)
    adapter=ParentPTv3(width=16,checkpointing=False)
    captured={}
    class FeaturesOnly(torch.nn.Module):
        def forward(self,data):
            captured.update(data); return SimpleNamespace(feat=data['feat']*2)
    adapter.net=FeaturesOnly()
    feat=torch.randn(3,16)
    out=adapter(feat,coords,g.centers(coords).float(),torch.zeros(1,32),g.parent_resolution)
    assert torch.equal(out,feat*2)
    assert torch.equal(captured['grid_coord'],coords[torch.argsort(ids)].int())
    assert captured['sparse_shape']==[257,257,33]
