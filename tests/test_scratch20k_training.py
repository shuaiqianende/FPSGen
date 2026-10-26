import ast,copy
from pathlib import Path
import pytest
import torch
from torch import nn
from fpsgen.hier_occ_flow import scratch20k_training as s
from fpsgen.hier_occ_flow.meanflow_math import meanflow_loss

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/research_v2/hier_occ_flow/scratch20k_stage1_meanflow_gpu2.yaml'


def test_registered_budget_and_precise41():
    cfg,g=s.load_config(CONFIG)
    assert cfg['training']['successful_frame_exposures']==20000 and cfg['training']['batch_per_gpu']==2
    assert cfg['validation']['frames']==list(range(0,4001,100))
    assert cfg['initialization_mode']=='scratch' and not any(k.startswith('source_stage') for k in cfg)


def test_no_pretrained_loader_and_both_encoder_flow_trainable(monkeypatch):
    cfg,g=s.load_config(CONFIG)
    class Tiny(nn.Module):
        def __init__(self,*a,**k):
            super().__init__();self.encoder=nn.Linear(3,4);self.flow=nn.Linear(4,1)
    monkeypatch.setattr(s,'Stage1DirectOccupancy',Tiny)
    monkeypatch.setattr(s,'MeanFlowStage1',lambda m:m)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Scratch constructor must not load checkpoint'))
    s.seed_all(20261032);a=s.ScratchSystem(cfg,g,'cpu')
    s.seed_all(20261032);b=s.ScratchSystem(cfg,g,'cpu')
    assert s.exact_state(a.state_dict(),b.state_dict())
    assert a.initialization['source_exposures']==0 and not a.initialization['pretrained_checkpoint_loaded']
    assert all(p.requires_grad for p in a.stage1.encoder.parameters()) and all(p.requires_grad for p in a.stage1.flow.parameters())


def test_native_meanflow_gradient_through_trainable_condition():
    encoder=nn.Linear(3,4);flow=nn.Linear(4,1);xyz=torch.randn(2,3);features=encoder(xyz)
    x=torch.randn(2,1);v=torch.randn_like(x);t=torch.tensor([.2,.3]);end=torch.tensor([.6,.8])
    fn=lambda z,a,b:flow(features)+.1*z+(b-a)[:,None]
    loss,detail=meanflow_loss(fn,x,t,end,v,torch.ones_like(x,dtype=torch.bool))
    loss.backward()
    assert detail['jvp_calls']==1 and not detail['target'].requires_grad
    assert encoder.weight.grad.isfinite().all() and encoder.weight.grad.norm()>0
    assert flow.weight.grad.norm()>0


def test_from_scratch_source_does_not_call_weight_loader():
    module=ast.parse((ROOT/'fpsgen/hier_occ_flow/scratch20k_training.py').read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='ScratchSystem')
    calls=[n for n in ast.walk(cls) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('load','load_state_dict') for n in calls)
    assert not any(isinstance(n.func,ast.Name) and n.func.id in ('load_sources','load_stage2_source','make_encoder_free_copy') for n in calls)


def test_resume_cannot_relabel_adaptation_or_reset_budget():
    identity={'initialization_mode':'scratch'}
    saved=dict(identity=identity,initialization=dict(initialization_mode='scratch',pretrained_checkpoint_loaded=False),
        counts=dict(exposures=20,updates=10,cursor=20,attempts=10,amp_skips=0),scheduler=dict(last_epoch=10))
    assert s.verify_resume(saved,identity) is saved
    wrong=copy.deepcopy(saved);wrong['initialization']['pretrained_checkpoint_loaded']=True
    with pytest.raises(ValueError):s.verify_resume(wrong,identity)
    wrong=copy.deepcopy(saved);wrong['counts']['exposures']=0
    with pytest.raises(ValueError):s.verify_resume(wrong,identity)


def test_scene_equal_fm_never_has_meanflow_interval():
    cfg,g=s.load_config(ROOT/'configs/research_v2/hier_occ_flow/scratch20k_stage2_no_encoder_fm_gpu3.yaml')
    assert cfg['physical_gpu']==3 and cfg['arm']=='S2-FM-NoEncoder-Scratch20K'
    from fpsgen.hier_occ_flow.stage2_batching import scene_mean_mse
    pred=torch.tensor([[1.],[2.],[2.]],requires_grad=True);target=torch.zeros_like(pred);mask=torch.ones_like(pred,dtype=torch.bool)
    loss=scene_mean_mse(pred,target,mask,torch.tensor([0,1,1]));assert loss==2.5
    loss.backward();assert torch.isfinite(pred.grad).all()


def test_real_stage1_constructor_cannot_load_pretrained(monkeypatch):
    cfg,g=s.load_config(CONFIG)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Actual scratch Stage1 constructor opened a checkpoint'))
    s.seed_all(20261032);model=s.ScratchSystem(cfg,g,'cpu')
    assert model.initialization['pretrained_checkpoint_loaded'] is False
    assert model.initialization['source_exposures']==0
    assert all(p.requires_grad for p in model.stage1.encoder.parameters())
    assert all(p.requires_grad for p in model.stage1.flow.parameters())
    assert model.initialization['parameters']>19000000


def test_real_stage2_constructor_no_encoder_or_interval_or_checkpoint(monkeypatch):
    cfg,g=s.load_config(ROOT/'configs/research_v2/hier_occ_flow/scratch20k_stage2_no_encoder_fm_gpu3.yaml')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Actual scratch Stage2 constructor opened a checkpoint'))
    s.seed_all(20261032);model=s.ScratchSystem(cfg,g,'cpu')
    assert model.initialization['source_exposures']==0
    assert model.stage2.fuse[0].in_features==176
    assert not hasattr(model,'stage1') and not hasattr(model.stage2,'lidar')
    assert not any('interval' in n for n,p in model.named_parameters())
