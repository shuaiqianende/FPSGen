"""Differentiable sqrt-CD using exact GPU tree only for discrete NN indices.

Same piecewise derivative as the existing Teacher KeOps loss: nearest assignment
is held fixed in backward, coordinate norms/gathers are handled by PyTorch.
This does not differentiate through tree construction or nearest-index changes.
"""
import torch
from fpsgen.ops.gpu_kdtree import ExactGPUKDTree


def chamfer_sqrt_gpu_kdtree(predicted,target,*,extension):
    p,g=predicted.float(),target.float()
    with torch.no_grad():
        gt_tree=ExactGPUKDTree(g.detach(),extension=extension)
        _,pg=gt_tree.query(p.detach())
        del gt_tree
        pred_tree=ExactGPUKDTree(p.detach(),extension=extension)
        _,gp=pred_tree.query(g.detach())
        del pred_tree
    return .5*((p-g[pg]).norm(dim=-1).mean()+(g-p[gp]).norm(dim=-1).mean())
