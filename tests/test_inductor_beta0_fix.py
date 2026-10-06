import pytest
import torch

from scripts.evaluation.inductor_beta0_fix import safe_baddbmm


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('alpha',[0,1,2,-1])
def test_beta_zero_does_not_read_additive(value,alpha):
    left=torch.arange(12,dtype=torch.float32).reshape(1,3,4)
    right=torch.arange(8,dtype=torch.float32).reshape(1,4,2)
    additive=torch.full((1,3,2),value)
    actual=safe_baddbmm(additive,left,right,beta=0,alpha=alpha)
    expected=torch.baddbmm(additive,left,right,beta=0,alpha=alpha)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('beta',[1,2,-1])
def test_nonzero_beta_preserves_additive(beta):
    left=torch.ones(2,3,4)
    right=torch.ones(2,4,5)
    additive=torch.ones(2,3,5)
    torch.testing.assert_close(safe_baddbmm(additive,left,right,beta=beta,alpha=2),
                               torch.baddbmm(additive,left,right,beta=beta,alpha=2),rtol=0,atol=0)


def test_installer_updates_cached_selected_table(monkeypatch):
    from torch._inductor import decomposition
    from scripts.evaluation.inductor_beta0_fix import install
    operation=torch.ops.aten.baddbmm.default
    original=decomposition.decompositions[operation]
    decomposition.select_decomp_table()
    try:
        monkeypatch.setattr(torch,'__version__','2.0.1+cu117')
        install()
        assert decomposition.select_decomp_table()[operation] is safe_baddbmm
        install()
        assert decomposition.select_decomp_table()[operation] is safe_baddbmm
    finally:
        decomposition.decompositions[operation]=original
        decomposition.fast_random_decomps.cache_clear()
