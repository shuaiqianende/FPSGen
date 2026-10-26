import copy,json
from pathlib import Path
import pytest
import torch
from torch import nn
from fpsgen.hier_occ_flow import queued20k_training as q
from fpsgen.hier_occ_flow import scratch20k_training as base
from fpsgen.hier_occ_flow.encoder_adaptation import tensor_state_sha
from fpsgen.hier_occ_flow.stage1_network import OccupancyUNet
from scripts.research_v2.queued20k.run_followup_queue import completed

ROOT=Path(__file__).resolve().parents[1]

def cfg(name):return q.load_config(ROOT/'configs/research_v2/hier_occ_flow'/name)


def test_all_three_exact_scratch20k_configs_and_dependencies():
    for name in ['queued_s1_fm_scratch20k_gpu3.yaml','queued_s1_shortcut_scratch20k_gpu2.yaml','queued_s2_encoder_fm_scratch20k_gpu3.yaml']:
        c,g=cfg(name)
        assert c['initialization_mode']=='scratch' and c['training']['successful_frame_exposures']==20000 and c['training']['batch_per_gpu']==2
        assert c['validation']['frames']==list(range(0,4001,100)) and c['validation']['nfes']==[1,2,4,8]
        assert not any(k.startswith('source_stage') for k in c)


def test_shortcut_uses_two_advancing_half_steps_and_detaches_target():
    p=nn.Parameter(torch.tensor(.3));x=torch.tensor([[.2],[.4]]);t=torch.tensor([.1,.2]);h=torch.tensor([.2,.1]);calls=[]
    def fn(z,a,d):calls.append((z.clone(),a.clone(),d.clone()));return p*z+3*a[:,None]
    result=q.shortcut_target(fn,x,t,h,torch.ones_like(x,dtype=torch.bool))
    one=.3*x+3*t[:,None];mid=x+h[:,None]*one;two=.3*mid+3*(t+h)[:,None]
    assert torch.allclose(result,(one+two)/2)
    assert len(calls)==2 and torch.allclose(calls[1][1],t+h) and torch.allclose(calls[1][0],mid)
    assert not result.requires_grad and p.grad is None


def test_invalid_bootstrap_domain_rejected():
    with pytest.raises(ValueError):q.shortcut_target(lambda x,t,d:x,torch.ones(1,1),torch.tensor([.9]),torch.tensor([.1]),torch.ones(1,1,dtype=torch.bool))


def test_dyadic_training_does_not_consume_fm_source_rng():
    source=torch.Generator().manual_seed(7);before=source.get_state().clone();other=torch.Generator().manual_seed(8)
    time=torch.rand(5000,generator=source);after=source.get_state().clone()
    t,d,mask,h=q.shortcut_pair(time,other,[.5,.25,.125,.0625,.03125,.015625,.0078125])
    assert torch.equal(source.get_state(),after)
    assert ((t>=0)&(t+d<=1.000001)).all() and torch.equal(d[~mask],torch.zeros_like(d[~mask]))
    assert .22<float(mask.float().mean())<.28
    assert torch.allclose(t[mask]/h[mask],(t[mask]/h[mask]).round())


def test_d0_and_initial_step_embedding_preserve_fm_outputs_and_learn():
    original=OccupancyUNet('fm',widths=(8,8,8,8),activation_checkpointing=False)
    wrapped=q.StepConditionedStage1(copy.deepcopy(original))
    x=torch.randn(2,1,8,8,8);obs=torch.zeros_like(x);pyramid=[torch.randn(2,128,4,4,4),torch.randn(2,128,2,2,2),torch.randn(2,128,1,1,1)];t=torch.tensor([.1,.3])
    expected=original(x,obs,pyramid,t)
    assert torch.equal(expected,wrapped(x,obs,pyramid,t,torch.zeros_like(t)))
    value=wrapped(x,obs,pyramid,t,torch.tensor([.5,.25]));assert torch.equal(expected,value)
    value.square().mean().backward();last=wrapped.base.embedding.interval.net[-1]
    assert sum(float(p.grad.norm()) for p in last.parameters())>0


def test_actual_common_scratch_stage1_parameters_match_meanflow(monkeypatch):
    c,g=cfg('queued_s1_fm_scratch20k_gpu3.yaml');sc,_=cfg('queued_s1_shortcut_scratch20k_gpu2.yaml')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('No checkpoint for new Stage1'))
    base.seed_all(20261032);fm=q.QueueSystem(c,g,'cpu');digest=tensor_state_sha(q.common_stage1_state(fm));del fm
    base.seed_all(20261032);shortcut=q.QueueSystem(sc,g,'cpu');assert tensor_state_sha(q.common_stage1_state(shortcut))==digest;del shortcut
    mc,_=base.load_config(ROOT/'configs/research_v2/hier_occ_flow/scratch20k_stage1_meanflow_gpu2.yaml')
    base.seed_all(20261032);mf=base.ScratchSystem(mc,g,'cpu');assert tensor_state_sha(q.common_stage1_state(mf))==digest
    assert mf.initialization['initial_state_sha256']=='1e7837680e8100176761627264c188a1d0d351cc8a65dceab6038507257f4058'


def test_s2_shared_random_weights_match_no_encoder_without_loading_teacher(monkeypatch):
    c,g=cfg('queued_s2_encoder_fm_scratch20k_gpu3.yaml')
    class FakeFrozen:
        def __init__(self,*a):self.model=nn.Linear(1,1).eval().requires_grad_(False);self.audit={'unit_test_only':True,'sha256':'fixture'}
        def __deepcopy__(self,memo):return self
    monkeypatch.setattr(q,'FrozenEncoder',FakeFrozen)
    base.seed_all(20261032);model=q.QueueSystem(c,g,'cpu')
    a=model.initialization['shared_initialization']
    assert a['reference_initial_state_sha256']==c['matched_no_encoder_initial_state_sha256'] and a['shared_tensors_exact'] and a['fuse_common_columns_exact']
    assert model.stage2.fuse[0].in_features==240 and hasattr(model.stage2,'lidar')
    assert not any('frozen_condition' in n for n,p in model.named_parameters())
    ema=copy.deepcopy(model);assert ema.frozen_condition is model.frozen_condition
    before=copy.deepcopy(model.frozen_condition.model.state_dict());base.ema_update(ema,model)
    assert base.exact_state(before,model.frozen_condition.model.state_dict())


def test_queue_waits_until_both_training_and_exact41_quality(tmp_path):
    dep={'root':str(tmp_path),'arm':'fake','config_sha256':'cfg','source_sha256':'src'}
    assert completed(dep)[0] is False
    (tmp_path/'formal').mkdir();(tmp_path/'evaluation/seq08_41').mkdir(parents=True)
    summary={'status':'TRAINING_COMPLETE','counts':{'updates':10000,'exposures':20000},'identity':{'arm':'fake','config_sha256':'cfg','source_sha256':'src','initialization_mode':'scratch'}}
    (tmp_path/'formal/summary.json').write_text(json.dumps(summary))
    assert completed(dep)[0] is False
    p=tmp_path/'evaluation/seq08_41/results.json';p.write_text(json.dumps({'status':'EVALUATED','frames':41,'protocol':{'frames':list(range(0,4001,100))}}))
    assert completed(dep)[0] is True
    p.write_text(json.dumps({'status':'EVALUATED','frames':900,'protocol':{'frames':list(range(900))}}))
    with pytest.raises(ValueError):completed(dep)


def test_frozen_encoder_accepts_only_declared_scratch20k_source(tmp_path,monkeypatch):
    from fpsgen.hier_occ_flow.storage import file_sha
    c,g=cfg('queued_s2_encoder_fm_scratch20k_gpu3.yaml');spec=c['frozen_encoder'];source=tmp_path/'unit_encoder.pt';spec=dict(spec,checkpoint=str(source),source_sha256='unit-source');c=dict(c,frozen_encoder=spec)
    class TinySource(nn.Module):
        def __init__(self,*a,**kw):super().__init__();self.encoder=nn.Linear(3,4)
    monkeypatch.setattr(q,'Stage1DirectOccupancy',TinySource)
    init={'pretrained_checkpoint_loaded':False}
    identity=dict(initialization_mode='scratch',arm=spec['arm'],config_sha256=spec['config_sha256'],source_sha256='unit-source',geometry_sha256=g.sha256,seed=20261032,
        train_manifest_sha256=file_sha(Path(c['train_dataset'])/'manifest.json'),development_manifest_sha256=file_sha(Path(c['development_dataset'])/'manifest.json'))
    fixture=dict(identity=identity,counts={'updates':10000,'exposures':20000},initialization=init,
        ema={'stage1.encoder.weight':torch.ones(4,3),'stage1.encoder.bias':torch.ones(4)})
    torch.save(fixture,source);holder=q.FrozenEncoder(c,g,'cpu')
    assert not holder.model.training and all(not p.requires_grad for p in holder.model.parameters())
    assert torch.equal(holder.model.weight,torch.ones(4,3)) and holder.audit['source_exposures']==20000
    fixture['identity']['source_sha256']='wrong';torch.save(fixture,source)
    with pytest.raises(ValueError):q.FrozenEncoder(c,g,'cpu')
    fixture['identity']['source_sha256']='unit-source';fixture['initialization']['pretrained_checkpoint_loaded']=True;torch.save(fixture,source)
    with pytest.raises(ValueError):q.FrozenEncoder(c,g,'cpu')


def test_queue_does_not_bypass_failed_predecessor(tmp_path):
    root=tmp_path/'previous';(root/'formal').mkdir(parents=True);(root/'formal/failure.json').write_text('{}')
    dep={'root':str(root),'arm':'fake','config_sha256':'cfg','source_sha256':'src'}
    with pytest.raises(RuntimeError):completed(dep)


def test_queue_cli_does_not_shadow_stdlib_queue():
    import subprocess,sys
    r=subprocess.run([sys.executable,str(ROOT/'scripts/research_v2/queued20k/run_followup_queue.py'),'--help'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr
    assert '--plan' in r.stdout and '--gpu' in r.stdout
