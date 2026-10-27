import pytest
from scripts.research_v2.finalize_mfv2_interim import validate_rows


def full_rows():return [dict(frame=f,nfe=n) for f in range(0,4001,100) for n in (1,2,4,8)]


def test_all41_frames_and_four_nfes_are_required():
    validate_rows(full_rows())
    with pytest.raises(ValueError):validate_rows(full_rows()[:-1])


def test_duplicate_record_cannot_replace_a_missing_frame_nfe():
    r=full_rows();r[-1]=r[0].copy()
    with pytest.raises(ValueError):validate_rows(r)


def test_extra_unregistered_frame_cannot_be_called_frozen41():
    r=full_rows();r[-1]=dict(frame=4100,nfe=8)
    with pytest.raises(ValueError):validate_rows(r)
