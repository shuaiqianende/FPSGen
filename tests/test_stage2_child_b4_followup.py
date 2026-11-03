import pytest
import torch
from fpsgen.hier_occ_flow.child_local_state import ChildLocalStateEncoder
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy

def test_child_scratch_shared_initialization_and_zero_residual():
    geometry=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    models=[]
    for mode in ('none','position','position_attention'):
        torch.manual_seed(20261032)
        models.append(HierarchicalOccupancyModel(geometry,backbone='local',local_state_encoding=mode))
    original,position,attention=models
    for candidate in (position,attention):
        for name,tensor in original.state_dict().items():
            assert torch.equal(tensor,candidate.state_dict()[name]),name
        state=torch.randn(7,8);valid=torch.ones(7,8,dtype=torch.bool)
        assert torch.equal(candidate.child_state_encoder(state,valid),torch.zeros(7,64))
        candidate.child_state_encoder(state,valid).sum().backward()
        grad=candidate.child_state_encoder.output.weight.grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum())>0

def test_child_branch_mask_and_backward_cpu():
    geometry=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    for mode in ('position','position_attention'):
        module=ChildLocalStateEncoder(geometry,mode=mode)
        # Nonzero projection exercises the full branch, not only the zero output layer.
        torch.nn.init.normal_(module.output.weight,std=.01)
        state=torch.randn(5,8,requires_grad=True);valid=torch.ones(5,8,dtype=torch.bool);valid[0,1]=False
        out=module(state,valid);out.float().square().mean().backward()
        assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(state.grad).all())
        assert state.grad[0,1].item()==0

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for explicit FP16')
def test_explicit_fp16_child_branch_cuda():
    geometry=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    for mode in ('position','position_attention'):
        module=ChildLocalStateEncoder(geometry,mode=mode).cuda()
        torch.nn.init.normal_(module.output.weight,std=.01)
        apply_explicit_policy(module)
        state=torch.randn(32,8,device='cuda',dtype=torch.float16,requires_grad=True)
        valid=torch.ones_like(state,dtype=torch.bool);valid[0,1]=False
        with torch.autocast('cuda',dtype=torch.float16):out=module(state,valid)
        out.float().square().mean().backward()
        assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(state.grad).all())
        assert state.grad[0,1].item()==0
