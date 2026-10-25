import copy,importlib
from pathlib import Path
import pytest
import torch
from test_encoder_adaptation import Stage1,T
from fpsgen.hier_occ_flow.encoder_adaptation import EncoderAdaptationSystem,frozen_state_sha,parameter_sha,update_adaptation_ema
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import make_optimizer


def test_flow_only_keeps_encoder_and_ema_frozen():
    torch.manual_seed(20261032)
    model=EncoderAdaptationSystem(Stage1(),'S1-FM20K');ema=copy.deepcopy(model).eval().requires_grad_(False)
    frozen=frozen_state_sha(model);flow_before=parameter_sha(model.stage1.flow)
    assert not model.stage1.encoder.training and model.stage1.flow.training
    opt,sched=make_optimizer(model,T)
    assert [g['name'] for g in opt.param_groups]==['stage1.flow']
    xyz=[torch.randn(8,3),torch.randn(8,3)]
    loss,_=model.stage1_loss(xyz,torch.ones(2,1,2,2,2),torch.Generator().manual_seed(20262032))
    loss.backward()
    assert model.stage1.flow.conv.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.stage1.encoder.parameters())
    opt.step();sched.step();update_adaptation_ema(ema,model,.999)
    assert parameter_sha(model.stage1.flow)!=flow_before
    assert frozen_state_sha(model)==frozen_state_sha(ema)==frozen


def test_exact_snapshot_comparison_detects_optimizer_and_rng_changes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/research_v2/hier_occ_flow'))
    exact=importlib.import_module('verify_stage1_flow_control').exact_state
    value={'optimizer':{'step':torch.tensor(3.),'moment':torch.tensor([.01,.02])},'rng':torch.tensor([1,2,3],dtype=torch.uint8)}
    assert exact(value,copy.deepcopy(value))
    other=copy.deepcopy(value);other['optimizer']['moment'][0]+=.001
    assert not exact(value,other)
    other=copy.deepcopy(value);other['rng'][0]=4
    assert not exact(value,other)


def test_gpu0_is_opt_in_and_does_not_change_default_cards(monkeypatch):
    from fpsgen.hier_occ_flow import adaptation_gpu_slot as slot
    monkeypatch.setenv('DESESCENE_SHARED_ADAPTATION','1');monkeypatch.delenv('DESESCENE_GPU0_FLOW_CONTROL',raising=False)
    with pytest.raises(ValueError):slot.adaptation_admission(0)
    monkeypatch.setenv('DESESCENE_GPU0_FLOW_CONTROL','1')
    calls=[];monkeypatch.setattr(slot,'gpu_preflight',lambda ids,**kw:calls.append((ids,kw)) or {'status':'AVAILABLE'})
    assert slot.adaptation_admission(0)['status']=='AVAILABLE'
    assert calls==[((0,),dict(minimum_free_gib=14.,allow_shared_compute=True))]
