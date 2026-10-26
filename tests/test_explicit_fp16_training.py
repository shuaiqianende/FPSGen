import copy
import pytest
import torch
from torch import nn
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy,FP32Masters,update_fp32_ema
from fpsgen.hier_occ_flow.explicit_fp16_replay import compare


def test_actual_half_parameters_keep_pristine_fp32_buffers():
    m=nn.Sequential(nn.Linear(3,4),nn.LayerNorm(4));m.register_buffer('frequencies',torch.tensor([1.000123,2.000234]));m.register_buffer('mask',torch.tensor([True,False]));old=m.frequencies.clone()
    report=apply_explicit_policy(m)
    assert all(p.dtype==torch.float16 for p in m.parameters()) and report['parameter_dtypes']==['torch.float16']
    assert torch.equal(m.frequencies,old) and m.frequencies.dtype==torch.float32 and m.mask.dtype==torch.bool


def test_norm_parameters_half_statistics_float32_and_gradient():
    m=nn.GroupNorm(2,4);reference=copy.deepcopy(m);apply_explicit_policy(m)
    x=torch.randn(2,4,3,3).half().requires_grad_(True);y=m(x)
    assert y.dtype==torch.float16 and torch.allclose(y.float(),reference(x.float()),atol=.002,rtol=.002)
    y.float().square().mean().backward();assert m.weight.grad.dtype==torch.float16 and torch.isfinite(m.weight.grad).all()


def test_batch_norm_running_stats_remain_fp32_and_update_once():
    m=nn.BatchNorm1d(4);ref=copy.deepcopy(m);apply_explicit_policy(m);x=torch.randn(8,4).half();y=m(x);z=ref(x.float())
    assert y.dtype==torch.float16 and m.running_mean.dtype==torch.float32 and int(m.num_batches_tracked)==1
    assert torch.equal(m.running_mean,ref.running_mean) and torch.equal(m.running_var,ref.running_var)
    assert torch.allclose(y.float(),z,atol=.002,rtol=.002)


def test_fp32_master_updates_below_half_ulp_are_not_lost():
    m=nn.Linear(1,1,bias=False);m.weight.data.fill_(1.);initial=copy.deepcopy(m.state_dict());masters=FP32Masters(m,initial);apply_explicit_policy(m)
    opt=torch.optim.SGD(masters.optimizer_groups([{'params':m.parameters(),'lr':1e-5}]))
    for _ in range(100):
        m.zero_grad();m.weight.float().sum().backward();masters.copy_gradients();opt.step();masters.copy_to_network()
    assert masters.params['weight'].dtype==torch.float32 and float(masters.params['weight'])<.9991
    assert torch.equal(m.weight,masters.params['weight'].half())


def test_master_adam_and_ema_stay_fp32_with_half_runtime():
    m=nn.Linear(1,1);initial=copy.deepcopy(m.state_dict());masters=FP32Masters(m,initial);apply_explicit_policy(m);ema=copy.deepcopy(m).eval();ema_state=copy.deepcopy(initial)
    opt=torch.optim.AdamW(masters.optimizer_groups([{'params':m.parameters(),'lr':.0002}]))
    sum(p.float().sum() for p in m.parameters()).backward();masters.copy_gradients();opt.step();masters.copy_to_network();update_fp32_ema(ema_state,ema,m,masters)
    assert all(v.dtype==torch.float32 for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
    assert all(v.dtype==torch.float32 for v in ema_state.values()) and all(p.dtype==torch.float16 for p in ema.parameters())
    saved=copy.deepcopy(masters.state_dict());masters.params['weight'].data.add_(1);masters.load_state_dict(saved);masters.copy_to_network();assert torch.equal(m.weight,saved['weight'].half())


def test_replay_requires_rng_exact_and_rejects_large_parameter_change():
    a={'model':{'w':torch.ones(2,dtype=torch.float16)},'master_parameters':{'w':torch.ones(2)},'rng':{'state':torch.tensor([1,2],dtype=torch.uint8)}}
    assert compare(a,copy.deepcopy(a))['status']=='PASS'
    b=copy.deepcopy(a);b['model']['w'][0]=1.1;assert compare(a,b)['status']=='FAIL'
    b=copy.deepcopy(a);b['rng']['state'][0]=3;assert compare(a,b)['status']=='FAIL'


def test_pipeline_promotes_existing_gate_and_pins_before_launch(tmp_path,monkeypatch):
    import json,sys
    from scripts.research_v2.explicit_fp16 import worker
    (tmp_path/'engineering').mkdir();(tmp_path/'engineering/result.json').write_text(json.dumps({'status':'GPU_PASS'}))
    monkeypatch.setattr(worker,'load_config',lambda p:({'output_root':str(tmp_path),'arm':'unit','physical_gpu':2},None))
    monkeypatch.setattr(worker,'own_source_sha',lambda:'unit-source');monkeypatch.setattr(worker,'config_sha',lambda c:'unit-config')
    class Child:
        pid=123
        def wait(self):return 1
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**k:Child());monkeypatch.setattr(sys,'argv',['worker','--config','unused.yaml','--mode','pipeline'])
    with pytest.raises(RuntimeError,match='smoke exited1'):worker.main()
    assert json.loads((tmp_path/'pipeline_launch.json').read_text())['source_sha256']=='unit-source'
