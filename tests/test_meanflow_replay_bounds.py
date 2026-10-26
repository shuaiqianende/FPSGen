import copy
import torch
from fpsgen.hier_occ_flow.meanflow_replay_bounds import compare_native_replay


def state():
    return dict(model={'w':torch.ones(3)},ema={'w':torch.ones(3)},
        optimizer={'state':{'0':{'step':torch.tensor(1.),'exp_avg':torch.ones(3)*.01,'exp_avg_sq':torch.ones(3)*.0001}}},
        gradient={'w':torch.ones(3)*.1},rng={'cuda':torch.tensor([1,2,3],dtype=torch.uint8)},
        interval_rng=torch.tensor([5,6],dtype=torch.uint8),loss=1.,gradient_norm=.2)


def test_small_cuda_float_drift_can_be_numeric_replay_but_never_exact():
    a=state();b=copy.deepcopy(a);b['model']['w'][0]+=1e-7;b['gradient']['w'][1]+=1e-6
    r=compare_native_replay(a,b)
    assert r['status']=='PASS' and not r['exact']


def test_rng_changed_adam_step_changed_and_large_gradient_are_rejected():
    for mutate in (lambda x:x['rng']['cuda'].add_(1),
                   lambda x:x['optimizer']['state']['0']['step'].add_(1),
                   lambda x:x['gradient']['w'].add_(.01)):
        a=state();b=copy.deepcopy(a);mutate(b)
        assert compare_native_replay(a,b)['status']=='FAIL'


def test_nonfinite_and_missing_state_never_pass():
    a=state();b=copy.deepcopy(a);b['model']['w'][0]=float('nan')
    assert compare_native_replay(a,b)['status']=='FAIL'
    b=copy.deepcopy(a);del b['interval_rng']
    assert compare_native_replay(a,b)['status']=='FAIL'
