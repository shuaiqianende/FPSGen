import copy
import hashlib
import json
import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.stage1_frame_budget import final_update_for_frames,learning_rate_multiplier,validate_frame_budget,verify_budget_transition


@pytest.mark.parametrize('updates,frames,expected',[(0,0,25000),(18550,37100,34275),(20000,42900,34275),(34275,100000,34275)])
def test_mixed_batch_successful_frame_accounting(updates,frames,expected):
    assert final_update_for_frames(updates,frames,4,100000)==expected


def test_frame_budget_does_not_count_overflow_attempts_and_stops_exactly():
    update,frames,attempts=20000,42900,20013
    terminal=final_update_for_frames(update,frames,4,100000)
    while update<terminal:
        attempts+=1
        if attempts%137==0:continue
        update+=1;frames+=4
    assert update==34275 and frames==100000 and attempts>update
    with pytest.raises(ValueError):final_update_for_frames(update,frames+4,4,100000)
    with pytest.raises(ValueError):final_update_for_frames(20000,42901,4,100000)


def test_last_successful_optimizer_update_reaches_exact_budget(tmp_path):
    from fpsgen.hier_occ_flow.trainer import atomic_checkpoint
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    updates,frames=34274,99996;terminal=final_update_for_frames(updates,frames,4,100000)
    calls=0
    while updates<terminal:
        optimizer.zero_grad();loss=model(torch.ones(4,2)).square().mean();loss.backward();optimizer.step()
        calls+=1;updates+=1;frames+=4
    atomic_checkpoint(tmp_path/'final.pt',{'successful_updates':updates,'successful_frame_exposures':frames,'optimizer':optimizer.state_dict()})
    result=torch.load(tmp_path/'final.pt')
    assert calls==1 and result['successful_updates']==34275 and result['successful_frame_exposures']==100000


def configuration():
    return {'scope':'formal','training':{'successful_updates':34275,'successful_frame_exposures':100000,
        'batch_per_gpu':2,'physical_gpus':[2,3],'warmup_updates':2000,'lr_encoder':1e-4,'lr_flow':2e-4},
        'budget_transition':{'source_updates':20000,'source_successful_frames':42900,
            'anchor_lr_factor':learning_rate_multiplier(20000,{'successful_updates':100000,'warmup_updates':2000}),
            'previous_update_budget':100000,'lr_policy':'continuous_cosine_to_frame_budget'}}


def test_cosine_continues_without_lr_jump_and_finishes_at_actual_budget():
    cfg=configuration();validate_frame_budget(cfg)
    anchor=cfg['budget_transition']['anchor_lr_factor']
    assert learning_rate_multiplier(20000,cfg['training'],cfg['budget_transition'])==anchor
    rates=[learning_rate_multiplier(u,cfg['training'],cfg['budget_transition']) for u in range(20000,34276)]
    assert rates[-1]==pytest.approx(0.,abs=1e-15) and np.all(np.diff(rates)<=0)
    assert rates[1]/rates[0]>.999999
    bad=copy.deepcopy(cfg);bad['training']['successful_updates']=25000
    with pytest.raises(ValueError,match='Mixed-batch'):validate_frame_budget(bad)


def test_budget_revision_rejects_other_protocol_changes(tmp_path):
    cfg=configuration();cfg['engineering_gate']='new';old=copy.deepcopy(cfg)
    old.pop('budget_transition');old['training'].pop('successful_frame_exposures');old['training']['successful_updates']=100000;old['engineering_gate']='old'
    source=tmp_path/'config.json';source.write_text(json.dumps(old));checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'own-checkpoint-unit-fixture')
    cfg['budget_transition'].update(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_config=str(source))
    identity={'experiment':'occ','objective':'fm','geometry_sha256':'384','train_manifest_sha256':'train','development_manifest_sha256':'dev','world_size':2,'output':str(tmp_path),'config_sha256':hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()}
    a=cfg['budget_transition']['anchor_lr_factor']
    value={'identity':identity,'initialization_mode':'scratch','parameter_sync_pass':True,'successful_updates':20000,
        'sampler':{'successful_frame_exposures':42900},'optimizer':{'param_groups':[{'lr':1e-4*a},{'lr':2e-4*a}]}}
    assert verify_budget_transition(value,identity,cfg,checkpoint) is value
    bad=copy.deepcopy(cfg);bad['training']['lr_encoder']=2e-4
    with pytest.raises(ValueError,match='anchor'):verify_budget_transition(value,identity,bad,checkpoint)
    bad=copy.deepcopy(cfg);bad['extra_loss']=1
    with pytest.raises(ValueError,match='only stop budget'):verify_budget_transition(value,identity,bad,checkpoint)
    bad=copy.deepcopy(cfg);bad['budget_transition']['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError,match='SHA mismatch'):verify_budget_transition(value,identity,bad,checkpoint)
