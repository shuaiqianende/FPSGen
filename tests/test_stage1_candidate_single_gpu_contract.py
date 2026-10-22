from pathlib import Path
import yaml

from fpsgen.hier_occ_flow.stage1_training import validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs/research_v2/hier_occ_flow'


def load_config():
    path = CONFIG_DIR / 'fidelity_v2_single_gpu_e2_asymmetric3d_20k_gpu2.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['geometry'] = yaml.safe_load((CONFIG_DIR / cfg['geometry']).read_text())
    return cfg


def test_e2_candidate_is_registered_for_single_gpu_20k():
    cfg = load_config()
    geometry = validate_config(cfg)
    assert geometry.parent_resolution == (384, 384, 32)
    assert cfg['training']['physical_gpus'] == [2]
    assert cfg['training']['batch_per_gpu'] == 2
    assert cfg['training']['successful_updates'] == 10000
    assert cfg['training']['successful_frame_exposures'] == 20000
    assert cfg['model']['encoder_variant'] == 'asymmetric3d'


def test_e1_candidate_is_registered_for_single_gpu_20k():
    path = CONFIG_DIR / 'fidelity_v2_single_gpu_e1_bottleneck96_20k_gpu3.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['geometry'] = yaml.safe_load((CONFIG_DIR / cfg['geometry']).read_text())
    geometry = validate_config(cfg)
    assert geometry.parent_resolution == (384, 384, 32)
    assert cfg['training']['physical_gpus'] == [3]
    assert cfg['training']['batch_per_gpu'] == 2
    assert cfg['training']['successful_updates'] == 10000
    assert cfg['training']['successful_frame_exposures'] == 20000
    assert cfg['model']['encoder_variant'] == 'bottleneck96'
    assert cfg['model']['dense_fidelity']['internal_channels'] == 96


def test_candidate_identity_cannot_run_unregistered_variant_or_budget():
    cfg = load_config()
    cfg['model']['encoder_variant'] = 'lightweight_skip'
    try:
        validate_config(cfg)
    except ValueError as error:
        assert 'E1/E2' in str(error)
    else:
        raise AssertionError('Candidate identity accepted an unregistered architecture')

    cfg = load_config()
    cfg['training']['successful_frame_exposures'] = 10000
    try:
        validate_config(cfg)
    except ValueError as error:
        assert '20K exposures' in str(error)
    else:
        raise AssertionError('Candidate identity accepted an unregistered exposure budget')
