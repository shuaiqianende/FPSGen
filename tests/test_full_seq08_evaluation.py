import numpy as np
import pytest

from scripts.evaluation.evaluate_full_seq08_seed42 import archive_incomplete, bootstrap, validate_full_result
from scripts.evaluation.evaluate_completed_bev_study import save


def test_paired_bootstrap_keeps_delta_sign():
    result = bootstrap(np.ones(20)*.03)
    assert result['mean_delta_fpsgen_minus_baseline'] == pytest.approx(.03)
    assert result['ci95'] == pytest.approx([.03,.03])


def test_subset_not_accepted_as_full_sequence(tmp_path):
    save(tmp_path/'summary.json', dict(condition='100',gt_dir='gt_possion',frame_stride=1,
         generation_seeds=[42],samples_per_frame=1,bev_steps=10,guidance_scale=2,
         runtime_precision='fp16',dense_backend='inductor',checkpoint_global_step=299000,
         checkpoint_epoch=500,layout_source='literal_all_zero_no_semantic_layout',num_frames=200))
    (tmp_path/'frame_manifest.txt').write_text('\n'.join(f'{i:06d}' for i in range(200)))
    with pytest.raises(ValueError,match='ALL4071'):
        validate_full_result(tmp_path,2)


def test_wrong_generation_seed_rejected_before_metrics(tmp_path):
    save(tmp_path/'summary.json',dict(checkpoint_epoch=500,generation_seeds=[20261001]))
    with pytest.raises(ValueError,match='evaluation mismatch'):
        validate_full_result(tmp_path,2)


def test_resume_preserves_failed_artifacts(tmp_path):
    directory=tmp_path/'FPSGen/cfg1'
    directory.mkdir(parents=True)
    (directory/'failure.json').write_text('original failure')
    (tmp_path/'FPSGen_cfg1.log').write_text('original traceback')
    archive=archive_incomplete(tmp_path,'FPSGen',1)
    from pathlib import Path
    archive=Path(archive)
    assert (archive/'result/failure.json').read_text()=='original failure'
    assert (archive/'FPSGen_cfg1.log').read_text()=='original traceback'
    assert not directory.exists()
    assert archive_incomplete(tmp_path,'FPSGen',1) is None


def test_resume_never_archives_completed_results(tmp_path):
    directory=tmp_path/'FPSGen/cfg2'
    directory.mkdir(parents=True)
    (directory/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='Completed result'):
        archive_incomplete(tmp_path,'FPSGen',2)
    assert (directory/'summary.json').read_text()=='{}'


def test_backend_exception_must_be_explicit(tmp_path):
    save(tmp_path/'summary.json',dict(condition='100',gt_dir='gt_possion',frame_stride=1,
         generation_seeds=[42],samples_per_frame=1,bev_steps=10,guidance_scale=1,
         runtime_precision='fp16',dense_backend='eager',checkpoint_global_step=299000,
         checkpoint_epoch=499,layout_source='literal_all_zero_no_semantic_layout',num_frames=200))
    (tmp_path/'frame_manifest.txt').write_text('\n'.join(f'{i:06d}' for i in range(200)))
    with pytest.raises(ValueError,match='evaluation mismatch'):
        validate_full_result(tmp_path,1)
    # Explicit eager is accepted as a backend, but a subset is still rejected.
    with pytest.raises(ValueError,match='ALL4071'):
        validate_full_result(tmp_path,1,dense_backend='eager')
