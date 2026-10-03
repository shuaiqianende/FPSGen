import copy
from pathlib import Path
import yaml


def test_formal_protocol_and_pilot_have_identical_architecture_and_loader():
    root=Path("configs/research_v2/formal_bev_compare")
    from scripts.run_formal_bev_compare import selected_config_path
    data=None
    for method in ("fpsgen_legacy","dic_s","sid2_s","dip_s16"):
        formal=yaml.safe_load(selected_config_path(method).read_text())
        pilot=yaml.safe_load(selected_config_path(method,pilot=True).read_text())
        assert formal["train"]["n_gpus"]==2 and formal["train"]["batch_size"]==16
        assert formal["train"]["max_epoch"]==500
        assert formal["train"]["checkpoint_every_n_epochs"]==50
        assert formal["data"]["gt_dir"]=="gt_possion" and formal["data"]["num_points"]==180000
        assert "08" not in formal["data"]["train"]
        assert formal["runtime"]["precision"]=="fp16"
        assert formal['runtime']['grad_scaler_init_scale']==8192
        if method == 'dic_s':
            assert not formal['runtime']['compile_dense']
            assert formal['runtime']['activation_checkpointing']
            assert formal['runtime']['backend_exception']
            assert formal['model']['condition']['gate_init']==.1
        else:
            assert formal['runtime']['compile_dense']
        assert formal["runtime"]["throughput_mode"]=="lightweight"
        assert formal["runtime"]["train_vis_interval"]==0
        assert data is None or data==formal["data"]
        data=formal["data"]
        for cfg in (formal,pilot):
            cfg.pop("experiment")
            cfg["train"].pop("max_epoch")
            for key in ("formal_manifest_dir","throughput_csv"):
                cfg["runtime"].pop(key)
        assert formal==pilot


def test_parameter_partition_does_not_double_count_legacy_frontend():
    from fpsgen.models.bev_backbones.factory import build_bev_backbone
    from fpsgen.utils.formal_bev import parameter_counts
    model=build_bev_backbone({"model":{"backbone":"legacy"}})
    counts=parameter_counts(model)
    assert counts["core"]+counts["condition"]+counts["pointpillar"]==counts["total"]
    assert counts["condition"]>0


def test_scaler_skip_monitor_uses_actual_overflow_flags_and_flushes_once(tmp_path):
    import torch
    from types import SimpleNamespace
    from fpsgen.utils.formal_bev_callbacks import FormalBEVRunCallback
    callback=FormalBEVRunCallback({"formal_manifest_dir":str(tmp_path)})
    callback.scaler=SimpleNamespace(_found_inf_per_device=lambda optimizer:{"cuda:0":torch.tensor(0.)})
    callback.on_before_optimizer_step(None,None,None,0)
    callback.scaler._found_inf_per_device=lambda optimizer:{"cuda:0":torch.tensor(1.)}
    callback.on_before_optimizer_step(None,None,None,0)
    assert callback.skips==0
    callback.collect_skips()
    assert callback.skips==1 and not callback.inf_flags
    callback.collect_skips()
    assert callback.skips==1


def test_preflight_checks_unscaled_optimizer_hook():
    import torch
    from types import SimpleNamespace
    from fpsgen.utils.training_callbacks import FiniteTrainingPreflightCallback
    condition = torch.nn.Linear(2, 2)
    model = torch.nn.Module()
    model.condition_encoder = condition
    module = torch.nn.Module()
    module.model = model
    for parameter in condition.parameters():
        parameter.grad = torch.ones_like(parameter)
    callback = FiniteTrainingPreflightCallback({})
    callback.on_before_optimizer_step(SimpleNamespace(), module, None, 0)
    assert callback.condition_gradient_seen
    condition.weight.grad[0, 0] = float('inf')
    import pytest
    with pytest.raises(RuntimeError, match='non-finite model gradients'):
        callback.on_before_optimizer_step(SimpleNamespace(), module, None, 0)


def test_bootstrap_metric_direction_and_frame_pairing():
    from scripts.summarize_formal_bev_compare import paired_bootstrap
    high=paired_bootstrap([.8,.9],[.7,.8],resamples=50)
    assert high["fraction_ours_better"]==1 and high["better_delta_sign"]=="positive"
    low=paired_bootstrap([.1,.2],[.3,.4],resamples=50,lower_is_better=True)
    assert low["fraction_ours_better"]==1 and low["better_delta_sign"]=="negative"


def test_source_fingerprint_includes_untracked_code(tmp_path):
    from fpsgen.utils.formal_bev import source_fingerprint
    (tmp_path / 'fpsgen').mkdir()
    code = tmp_path / 'fpsgen/local.py'
    code.write_text('answer = 42\n')
    first = source_fingerprint(tmp_path)
    code.write_text('answer = 43\n')
    second = source_fingerprint(tmp_path)
    assert first['sha256'] != second['sha256']
    assert 'fpsgen/local.py' in first['files']


def test_frozen_b200_covers_entire_sequence_with_uniform_unique_indices():
    import numpy as np
    frames = Path('configs/research_v2/formal_bev_compare/seq08_b200.txt').read_text().splitlines()
    assert len(frames) == len(set(frames)) == 200
    indices = np.asarray([int(f) for f in frames])
    assert indices[0] == 0 and indices[-1] == 4070
    assert set(np.diff(indices)) == {20, 21}


def test_final_summary_rejects_pilots_and_wrong_generation_protocol():
    import pytest
    from scripts.summarize_formal_bev_compare import validate_final_summary
    summary = dict(checkpoint_epoch=499, gt_dir='gt_possion', condition='100',
                   num_frames=200, samples_per_frame=3, bev_steps=10,
                   guidance_scale=2, generation_seeds=[20261001, 20261002, 20261003])
    validate_final_summary(summary, 2)
    with pytest.raises(ValueError, match='epoch500'):
        validate_final_summary(dict(summary, checkpoint_epoch=9), 2)
    with pytest.raises(ValueError, match='frozen protocol'):
        validate_final_summary(dict(summary, bev_steps=20), 2)


def test_formal_worker_propagates_failure_without_running_next_command(tmp_path):
    import json
    import sys
    from scripts.run_formal_bev_stage import run_stage
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(dict(directory=str(tmp_path), cwd=str(tmp_path), environment={},
                         command=[sys.executable, '-c', 'print("loss=0.25"); raise SystemExit(7)'])))
    assert run_stage(spec)==7
    assert json.loads((tmp_path / 'worker_result.json').read_text())['returncode']==7
    assert 'loss=0.25' in (tmp_path / 'console.log').read_text()


def test_generated_stage_config_cannot_overwrite_previous_evidence(tmp_path, monkeypatch):
    import pytest
    from scripts import run_formal_bev_compare as campaign
    monkeypatch.setattr(campaign, 'OUTPUT', tmp_path)
    base = dict(experiment={'id': 'test'},train={},runtime={'compile_dense':False})
    path = campaign.stage_config(base, 'example')
    original = path.read_text()
    with pytest.raises(FileExistsError):
        campaign.stage_config(base, 'example', compile_dense=True)
    assert path.read_text() == original


def test_pipeline_preflight_stops_checks_without_stopping_training():
    import torch
    from types import SimpleNamespace
    from fpsgen.utils.training_callbacks import FiniteTrainingPreflightCallback
    callback=FiniteTrainingPreflightCallback({'preflight_steps':1,'preflight_only':False})
    trainer=SimpleNamespace(should_stop=False)
    callback.on_train_batch_end(trainer,None,torch.tensor(1.),None,0)
    # Later steps belong to the throughput gate, not per-step synchronized
    # preflight checks. They must not inflate the validation-window counter.
    callback.on_train_batch_end(trainer,None,torch.tensor(float('nan')),None,1)
    assert callback.steps_seen==1 and not trainer.should_stop
