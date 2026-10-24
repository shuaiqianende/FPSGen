"""Failure and comparison guards, without starting a GPU process."""
import copy,importlib,json,threading
from pathlib import Path
import pytest
import numpy as np
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/research_v2/hier_occ_flow'))
    return importlib.import_module('aggregate_encoder_adaptation'),importlib.import_module('run_encoder_adaptation_phases')


def test_health_rejects_extra_budget_and_wrong_exposures(modules,tmp_path):
    _,queue=modules
    data={'status':'COMPLETE_REQUESTED_BUDGET','successful_updates':100,'successful_frame_exposures':200,'attempts':101,'amp_skips':1,'freeze_violation':False}
    path=tmp_path/'summary.json';path.write_text(json.dumps(data));assert queue.health(tmp_path,100)['successful_updates']==100
    for key,value in [('successful_updates',10001),('successful_frame_exposures',201),('freeze_violation',True),('amp_skips',9)]:
        bad=dict(data,**{key:value});path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError,match='health'):queue.health(tmp_path,100)


def test_geometry_comparison_requires_identical_fixed_support_and_inputs(modules):
    agg,_=modules
    rows=[{'sequence':'00','frame':i,'input_sha256':str(i),'parent_ids_sha256':'fixed',
        'fine':dict(iou=.3,f1=.4,precision=.5,recall=.3),
        'continuous':dict(cd_m=.2,iou05=.4,iou02=.3,iou01=.2,jsd_bev=.01,jsd_3d=.02)} for i in range(900)]
    candidate=copy.deepcopy(rows);candidate[42]['parent_ids_sha256']='changed'
    with pytest.raises(ValueError,match='support'):agg.fine_comparison(rows,candidate,fixed_support=True)
    candidate[42]['input_sha256']='different'
    with pytest.raises(ValueError,match='Input'):agg.fine_comparison(rows,candidate,fixed_support=False)
    with pytest.raises(ValueError,match='900'):agg.fine_comparison(rows[:64],rows[:64],fixed_support=True)


def test_timing_requires_same_gpu_and_three_whole_rounds(modules,tmp_path):
    agg,_=modules
    for round_id in (1,2,3):
        folder=tmp_path/f'round{round_id}';folder.mkdir()
        for arm in ('EF0','EF1'):
            value={'rows':[dict(sequence='00',frame=i,input_sha256=str(i),generation_wall_ms=100 if arm=='EF0' else 90) for i in range(20)],
                'summary':{},'physical_gpu':2,'peak_allocated_gib':3,'peak_reserved_gib':4}
            (folder/(arm+'.json')).write_text(json.dumps(value))
    report=agg.timing_summary(tmp_path);assert report['paired_generation']['EF1']['speedup_established']
    path=tmp_path/'round3/EF1.json';bad=json.loads(path.read_text());bad['physical_gpu']=3;path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Same physical GPU'):agg.timing_summary(tmp_path)
    path.unlink()
    with pytest.raises(ValueError,match='Three rounds'):agg.timing_summary(tmp_path)


def test_admission_falls_back_to_idle_card_without_reusing_a_lease(modules,monkeypatch):
    _,module=modules;q=module.Queue.__new__(module.Queue)
    q.gpu_lock=threading.Lock();q.gpu_leases=set();q.intact=lambda:None
    monkeypatch.setattr(module,'gpu_preflight',lambda ids:{'status':'AVAILABLE' if ids==(3,) else 'BLOCKED'})
    with q.lease(2) as gpu:assert gpu==3 and q.gpu_leases=={3}
    assert not q.gpu_leases
    monkeypatch.setattr(module,'gpu_preflight',lambda ids:{'status':'AVAILABLE'})
    with q.lease(2) as gpu:
        with q.lease(2) as other:assert gpu==2 and other==3 and q.gpu_leases=={2,3}
    assert not q.gpu_leases


def test_resume_auxiliary_audit_includes_numpy_rng_and_optimizer_tensors(modules):
    verifier=importlib.import_module('verify_encoder_adaptation_gpu')
    state={'rng':{'numpy':('MT19937',np.array([1,2,2**32-1],dtype=np.uint32),3)},
        'optimizer':{'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.tensor([.1,.2])}}}}
    tensors,scalars=verifier.flatten_state(state)
    assert torch.equal(tensors['/rng/numpy/1'],torch.tensor([1,2,2**32-1]))
    assert '/optimizer/state/0/exp_avg' in tensors and scalars['/rng/numpy/2']==3


def test_maximum_pair_requires_complete_census_and_chooses_largest_real_scenes(modules,tmp_path):
    verifier=importlib.import_module('verify_encoder_adaptation_gpu')
    class Dataset:
        def __init__(self,frames):self.index=np.array([(b'00',i) for i in frames],dtype=[('sequence','S2'),('frame','i4')])
        def __getitem__(self,i):return {'sequence':'00','frame':int(self.index[i]['frame'])}
    for arm in ('position','attention'):(tmp_path/(arm+'_20k')).mkdir()
    pos=tmp_path/'position_20k/train_rank0.jsonl';att=tmp_path/'attention_20k/train_rank0.jsonl'
    pos.write_text(json.dumps({'batch_records':[dict(sequence='00',frame=1,tokens=350)]})+'\n');att.write_text('')
    census=tmp_path/'census.jsonl';census.write_text(json.dumps(dict(sequence='00',frame=3,stage1_threshold_parent_count=900))+'\n')
    train,dev=Dataset([1,2]),Dataset([3])
    with pytest.raises(ValueError,match='Complete'):verifier.stress_records(train,dev,tmp_path,census)
    att.write_text(json.dumps({'batch_records':[dict(sequence='00',frame=2,tokens=480)]})+'\n')
    records,selection=verifier.stress_records(train,dev,tmp_path,census)
    assert [r['frame'] for r in records]==[3,2] and [r['teacher_parent_tokens'] for r in selection]==[900,480]
