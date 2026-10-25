import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.meanflow_training import load_config,verify_resume,optimizer_for,update_ema
from fpsgen.hier_occ_flow.meanflow_math import meanflow_loss
from fpsgen.hier_occ_flow.meanflow_queue import predecessor_status
from fpsgen.hier_occ_flow.meanflow_result_analysis import grouped_interval
from fpsgen.hier_occ_flow.trainer import rng_state,restore_rng

ROOT=Path(__file__).resolve().parents[1]


def test_preregistered_two_independent_configs_and_strict_limits():
    for stage in (1,2):
        cfg,g=load_config(ROOT/f'configs/research_v2/hier_occ_flow/meanflow_s{stage}_20k.yaml')
        assert cfg['training']['successful_frame_exposures']==20000 and cfg['training']['batch_per_gpu']==2
        assert cfg['meanflow']['jvp_backend']=='torch.func.jvp' and cfg['meanflow']['adaptive_power']==0
        assert cfg['validation']['seq08_frames']==list(range(0,4001,100))


def test_checkpoint_cpu_native_jvp_optimizer_ema_rng_replay_with_interval_rng(tmp_path):
    torch.manual_seed(3);net=torch.nn.Linear(3,3);ema=copy.deepcopy(net).requires_grad_(False)
    cfg={'training':{'learning_rate':1e-5,'weight_decay':.01}};opt,sched=optimizer_for(net,cfg)
    gen=torch.Generator().manual_seed(10);interval=torch.Generator().manual_seed(20)
    device=torch.device('cpu')
    def snapshot():return copy.deepcopy(dict(model=net.state_dict(),ema=ema.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),rng=rng_state(gen,device),interval=interval.get_state()))
    def restore(v):
        net.load_state_dict(v['model']);ema.load_state_dict(v['ema']);opt.load_state_dict(v['opt']);sched.load_state_dict(v['sched'])
        restore_rng(v['rng'],gen,device);interval.set_state(v['interval'])
    def update():
        x=torch.randn(2,3,generator=gen);t=torch.rand(2,generator=gen);s=t+(1-t)*torch.rand(2,generator=interval);v=torch.randn(2,3,generator=gen)
        fn=lambda z,a,b:net(z)+a[:,None]+(b-a)[:,None]*net(z).tanh()
        opt.zero_grad();loss,_=meanflow_loss(fn,x,t,s,v,torch.ones_like(x,dtype=torch.bool),model=net)
        loss.backward();opt.step();sched.step();update_ema(ema,net,.999)
        return loss.detach().clone(),snapshot()
    update();torch.save(snapshot(),tmp_path/'state.pt');first,state_a=update()
    restore(torch.load(tmp_path/'state.pt'));second,state_b=update()
    assert torch.equal(first,second)
    for key in ('model','ema'):
        assert all(torch.equal(state_a[key][n],state_b[key][n]) for n in state_a[key])
    assert torch.equal(state_a['interval'],state_b['interval']) and torch.equal(state_a['rng']['source'],state_b['rng']['source'])


def test_resume_rejects_identity_budget_and_interval_rng_changes():
    identity={'arm':'S1-MF20K'};value=dict(identity=identity,counts=dict(updates=2,exposures=4,attempts=3,amp_skips=1,cursor=6,diagonal_exposures=3,finite_interval_exposures=1),scheduler={'last_epoch':2},interval_rng=torch.zeros(2))
    assert verify_resume(value,identity) is value
    for key,val in [('exposures',5),('cursor',4),('finite_interval_exposures',2)]:
        bad=copy.deepcopy(value);bad['counts'][key]=val
        with pytest.raises(ValueError):verify_resume(bad,identity)
    bad=copy.deepcopy(value);bad.pop('interval_rng')
    with pytest.raises(ValueError):verify_resume(bad,identity)
    with pytest.raises(ValueError):verify_resume(value,{'arm':'S2-MF20K'})


def test_queue_never_launches_from_supervisor_claim_without_eval_proof(tmp_path):
    state=tmp_path/'supervisor.json';proof=tmp_path/'evaluation.json'
    pred=[dict(name='old',status_file=str(state),required_status='COMPLETE',proofs=[dict(path=str(proof),equals={'frames':900,'status':'COMPLETE'})])]
    assert predecessor_status(pred)['status']=='WAITING_PREDECESSOR_EVAL'
    state.write_text(json.dumps({'status':'TRAINING'}));assert predecessor_status(pred)['status']=='WAITING_PREDECESSOR_EVAL'
    state.write_text(json.dumps({'status':'COMPLETE'}));assert predecessor_status(pred)['status']=='BLOCKED_PREDECESSOR'
    proof.write_text(json.dumps({'status':'COMPLETE','frames':899}));assert predecessor_status(pred)['status']=='BLOCKED_PREDECESSOR'
    proof.write_text(json.dumps({'status':'COMPLETE','frames':900}));assert predecessor_status(pred)['status']=='READY'
    state.write_text(json.dumps({'status':'FAILED_REVIEW_REQUIRED'}));assert predecessor_status(pred)['status']=='BLOCKED_PREDECESSOR'


def test_adaptive_weights_are_detached_and_scene_equal():
    w=torch.nn.Parameter(torch.tensor(2.));x=torch.tensor([[1.],[3.],[3.]])
    time=torch.tensor([.2,.3]);batch=torch.tensor([0,1,1]);valid=torch.ones_like(x,dtype=torch.bool)
    fn=lambda z,t,s:w*z
    loss,detail=meanflow_loss(fn,x,time,time,torch.zeros_like(x),valid,row_to_scene=batch,adaptive_power=.5)
    q=detail['per_scene'];expected=((q.detach()+.001).pow(-.5)*q).mean()
    assert torch.equal(loss,expected)
    assert torch.equal(torch.autograd.grad(loss,w,retain_graph=True)[0],torch.autograd.grad(expected,w)[0])


def test_grouped_ci_does_not_claim_independent_training_seeds():
    ci=grouped_interval([1.,1.,1.,1.],[('00',0),('00',0),('00',1),('01',0)],draws=20)
    assert ci==[1.,1.]
