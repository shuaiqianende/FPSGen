"""Inference-only GPU authorization and unchanged training/checkpoint identity."""
import importlib.util
from pathlib import Path
import pytest
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,config_sha

path=Path('scripts/research_v2/hier_occ_inference/export_stage2_snapshot.py')
spec=importlib.util.spec_from_file_location('stage2_snapshot_export',path)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_predicted_diagnostic_policy_does_not_change_training_config():
    cfg,g=load_stage2_config('configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_100kframes.yaml')
    before=config_sha(cfg);policy=entry.predicted_policy(cfg)
    assert policy['mode']=='predicted' and policy['selection']['threshold']==.5
    assert policy['selection']['overflow']=='error'
    assert cfg['candidates']['mode']=='oracle' and config_sha(cfg)==before
    value={'identity':{'experiment':'hs-occflow-stage2-384-v1','geometry_sha256':g.sha256,
        'config_sha256':before,'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256']},
        'initialization':'scratch_stage2','parameter_sync_pass':True}
    entry.validate_snapshot(value,cfg,g)
    value['identity']['config_sha256']='inference-policy-is-not-the-training-identity'
    with pytest.raises(ValueError,match='identity mismatch'):entry.validate_snapshot(value,cfg,g)


def test_gpu1_requires_explicit_visibility_and_idle_admission(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    with pytest.raises(ValueError,match='GPU1 only'):entry.admit_gpu1()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    def output(cmd,**kwargs):return '' if '--query-compute-apps=pid' in cmd else '24000, 0\n'
    monkeypatch.setattr(entry.subprocess,'check_output',output)
    assert entry.admit_gpu1()['training_gpu_policy_modified'] is False
    def busy(cmd,**kwargs):return '12345\n' if '--query-compute-apps=pid' in cmd else '24000, 0\n'
    monkeypatch.setattr(entry.subprocess,'check_output',busy)
    with pytest.raises(RuntimeError,match='busy'):entry.admit_gpu1()
