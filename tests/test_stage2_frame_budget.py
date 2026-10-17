"""Successful-frame budgets and an explicit user quality-policy change."""
import copy
import json
from pathlib import Path
import pytest
from fpsgen.hier_occ_flow.stage2_runtime import (
    load_stage2_config,frame_budget_plan,require_training_gate,direct_full_train_policy)
from fpsgen.hier_occ_flow.storage import file_sha


def test_frame_budget_keeps_actual_old_single_gpu_exposures():
    # 160 old B1 updates, then240 GB4 updates. Do not multiply the old160 by4.
    plan=frame_budget_plan(400,1120,100000,4)
    assert plan['remaining_successful_updates']==24720
    assert plan['target_successful_updates']==25120
    assert 1120+24720*4==100000


def test_frame_budget_continuation_and_restoration_do_not_reset_budget():
    initial=frame_budget_plan(700,2320,100000,4)
    restored=frame_budget_plan(720,2400,100000,4)
    assert initial['target_successful_updates']==restored['target_successful_updates']==25120
    assert frame_budget_plan(25120,100000,100000,4)['remaining_successful_updates']==0


@pytest.mark.parametrize('values',[(400,1121,100000,4),(400,100004,100000,4),(-1,0,100000,4),(0,0,100000,0)])
def test_invalid_or_unreachable_frame_budget_rejected(values):
    with pytest.raises(ValueError):frame_budget_plan(*values)


def fixture_policy(tmp_path):
    cfg,g=load_stage2_config(Path('configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_overfit.yaml'))
    cfg.pop('cohort_indices');cfg['training'].update(scope='formal',successful_frame_exposures=100000,successful_updates=25120)
    evidence=tmp_path/'failed_quality.json';evidence.write_text(json.dumps({'status':'FAIL_ORACLE_OVERFIT',
        'geometry_sha256':g.sha256,'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256']}))
    cfg['direct_training']={'authorization':'user_requested_100k_successful_frames','quality_gate_policy':'monitor_only',
        'previous_quality_evidence':str(evidence),'previous_quality_sha256':file_sha(evidence)}
    gate={'geometry_sha256':g.sha256,'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],
          **{k:True for k in ('full_scene_gpu','maximum_scene_gpu','ptv3_permutation','checkpointing_parity',
                              'ddp_resume','packed_batch2_gpu','packed_batch2_ddp_resume')},
          'oracle_fm_overfit':False,'pilot5000':False}
    path=tmp_path/'gate.json';path.write_text(json.dumps(gate));cfg['engineering_gate']=str(path)
    return cfg,g,path,gate


def test_explicit_user_request_records_failed_quality_without_faking_pass(tmp_path):
    cfg,g,path,gate=fixture_policy(tmp_path)
    assert require_training_gate(cfg,g)['oracle_fm_overfit'] is False
    decision=direct_full_train_policy(cfg,g)
    assert decision['actual_previous_quality_status']=='FAIL_ORACLE_OVERFIT'
    assert decision['engineering_checks_waived'] is False and decision['quality_pass_claimed'] is False
    assert json.loads(path.read_text())==gate


@pytest.mark.parametrize('missing',['full_scene_gpu','maximum_scene_gpu','ddp_resume','packed_batch2_gpu','packed_batch2_ddp_resume'])
def test_user_quality_decision_never_waives_gpu_or_restore_checks(tmp_path,missing):
    cfg,g,path,gate=fixture_policy(tmp_path);gate[missing]=False;path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='gate missing'):require_training_gate(cfg,g)


def test_no_implicit_quality_override_or_old_evidence_relabeling(tmp_path):
    cfg,g,path,gate=fixture_policy(tmp_path);old=copy.deepcopy(cfg);old.pop('direct_training')
    with pytest.raises(ValueError,match='gate missing'):require_training_gate(old,g)
    cfg['direct_training']['previous_quality_sha256']='changed'
    with pytest.raises(ValueError,match='Preserve'):require_training_gate(cfg,g)


def test_direct_request_cannot_be_used_for_predicted_or_small_cohort(tmp_path):
    cfg,g,_,_=fixture_policy(tmp_path)
    cfg['candidates']['mode']='predicted'
    with pytest.raises(ValueError,match='specific'):direct_full_train_policy(cfg,g)
    cfg['candidates']['mode']='oracle';cfg['cohort_indices']=[1,2]
    with pytest.raises(ValueError,match='specific'):direct_full_train_policy(cfg,g)
