from pathlib import Path

import copy
import yaml
import pytest

from fpsgen.hier_occ_flow.stage1_training import LONGRUN_SCOPE, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/research_v2/hier_occ_flow"


def _load(name):
    path = CONFIGS / name
    cfg = yaml.safe_load(path.read_text())
    cfg["geometry"] = yaml.safe_load((CONFIGS / cfg["geometry"]).read_text())
    return cfg


def test_registered_longrun_pair_is_matched_except_variant_and_gpu():
    r0 = _load("fidelity_v2_longrun_r0_gpu2.yaml")
    e3 = _load("fidelity_v2_longrun_e3_gpu3.yaml")
    assert validate_config(r0).parent_resolution == (384, 384, 32)
    assert validate_config(e3).parent_resolution == (384, 384, 32)
    assert r0["scope"] == e3["scope"] == LONGRUN_SCOPE
    assert r0["training"]["physical_gpus"] == [2]
    assert e3["training"]["physical_gpus"] == [3]
    for key in ("successful_updates", "successful_frame_exposures", "seed",
                "batch_per_gpu", "lr_encoder", "lr_flow", "warmup_updates"):
        assert r0["training"][key] == e3["training"][key]
    for key in ("loss", "flow", "validation", "train_dataset", "development_dataset"):
        assert r0[key] == e3[key]
    assert r0["model"]["encoder_variant"] == "original"
    assert e3["model"]["encoder_variant"] == "lightweight_skip"


@pytest.mark.parametrize("change", [
    lambda cfg: cfg["training"].update(successful_updates=24999),
    lambda cfg: cfg["training"].update(physical_gpus=[1]),
    lambda cfg: cfg["training"].update(batch_per_gpu=2),
    lambda cfg: cfg["model"].update(encoder_variant="bottleneck96"),
    lambda cfg: cfg.update(experiment="unregistered-run"),
])
def test_longrun_rejects_unregistered_budget_or_arm(change):
    cfg = _load("fidelity_v2_longrun_r0_gpu2.yaml")
    changed = copy.deepcopy(cfg)
    change(changed)
    with pytest.raises(ValueError):
        validate_config(changed)
