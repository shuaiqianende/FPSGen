from pathlib import Path

import yaml

from fpsgen.hier_occ_flow.stage1_training import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/research_v2/hier_occ_flow"


def _load(name):
    path = CONFIGS / name
    cfg = yaml.safe_load(path.read_text())
    cfg["geometry"] = yaml.safe_load((CONFIGS / cfg["geometry"]).read_text())
    return cfg


def test_registered_pilot_variants_validate_with_fixed_20k_budget():
    for name in ("omd3d_v1_r0_original.yaml", "omd3d_v1_r1_bottleneck.yaml",
                 "omd3d_v1_r2_omd3d.yaml"):
        cfg = _load(name)
        geometry = validate_config(cfg)
        assert geometry.parent_resolution == (384, 384, 32)
        assert cfg["training"]["successful_frame_exposures"] == 20000
        assert cfg["training"]["successful_updates"] == 5000


def test_non_original_variant_cannot_use_formal_or_general_scope():
    cfg = _load("omd3d_v1_r1_bottleneck.yaml")
    cfg["scope"] = "formal"
    try:
        validate_config(cfg)
    except ValueError as error:
        assert "architecture_pilot" in str(error)
    else:
        raise AssertionError("architecture variant escaped pilot scope")


def test_formal_frame_budget_still_requires_100k_registration():
    cfg = _load("omd3d_v1_r0_original.yaml")
    cfg["scope"] = "formal"
    cfg.pop("experiment")
    cfg["training"]["successful_frame_exposures"] = 20000
    try:
        validate_config(cfg)
    except ValueError as error:
        assert "100K" in str(error)
    else:
        raise AssertionError("pilot budget was accepted by formal validation")
