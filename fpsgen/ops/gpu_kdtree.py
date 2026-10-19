"""Exact, evaluation-only GPU KD-tree CD; GPU sorting/build/query, no padding.

No third-party install is needed. The CUDA extension is built explicitly into a
project-local cache. Tree splits cycle through XYZ and use GPU segmented sorting.
Bounding-box branch-and-bound visits every node that can contain a closer point.
"""
import math
from pathlib import Path
import torch

_EXTENSION = None


def load_extension(build_directory):
    """Compile only when explicitly called; build/cache stay inside FPSGen."""
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        directory = Path(build_directory).resolve()
        project = Path('/data-12/M2024-HWZ/FPSGen').resolve()
        if not directory.is_relative_to(project):
            raise ValueError('Build cache must stay inside FPSGen')
        directory.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load(name='fpsgen_exact_gpu_kdtree_v1',
            sources=[str(Path(__file__).with_name('gpu_kdtree_cuda.cu'))],
            build_directory=str(directory), extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)
    return _EXTENSION


def _validate(points):
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
        or points.device.type != 'cuda' or points.dtype != torch.float32
        or not bool(torch.isfinite(points).all())):
        raise ValueError('Nonempty finite CUDA FP32 [N,3] coordinates required')


class ExactGPUKDTree:
    """Balanced spatial index; all point-dependent operations execute on CUDA."""
    @torch.no_grad()
    def __init__(self, support, *, extension, leaf_size=32):
        _validate(support)
        if leaf_size < 8:
            raise ValueError('leaf_size must be >=8')
        self.extension = extension
        count = len(support)
        depth = max(0, (math.ceil(count/leaf_size)-1).bit_length())
        if depth > 28:
            raise ValueError('Tree depth exceeds bounded exact traversal stack')
        self.leaf_count = 1 << depth
        self.points = support.contiguous()
        self.original_ids = torch.arange(count, device=support.device)
        positions = self.original_ids
        # Segment boundaries floor(j*N/G) remain nested at all levels.
        for level in range(depth):
            groups = 1 << level
            group_ids = ((positions+1)*groups-1)//count
            bits = self.points[:, level % 3].contiguous().view(torch.int32).long()
            ordered_float = torch.where(bits < 0, ~bits, bits ^ 0x80000000) & 0xffffffff
            keys = (group_ids << 32) | ordered_float
            order = torch.argsort(keys)
            self.points = self.points[order].contiguous()
            self.original_ids = self.original_ids[order]
        self.boxes = extension.build_boxes(self.points, self.leaf_count)

    @torch.no_grad()
    def query(self, query):
        _validate(query)
        if query.device != self.points.device:
            raise ValueError('Query and tree must share the admitted CUDA device')
        return self.extension.query(query.contiguous(), self.points, self.original_ids,
                                    self.boxes, self.leaf_count)


@torch.no_grad()
def chamfer_kdtree(predicted, target, *, extension, leaf_size=32):
    """Same 0.5*(mean Euclidean P->GT + GT->P), in meters; no approximation."""
    target_tree = ExactGPUKDTree(target, extension=extension, leaf_size=leaf_size)
    forward2, _ = target_tree.query(predicted)
    del target_tree
    predicted_tree = ExactGPUKDTree(predicted, extension=extension, leaf_size=leaf_size)
    reverse2, _ = predicted_tree.query(target)
    a, b = float(forward2.sqrt().mean()), float(reverse2.sqrt().mean())
    if not math.isfinite(a+b):
        raise ValueError('Nonfinite GPU KD-tree CD')
    return dict(valid_cd=True, cd_m=.5*(a+b), pred_to_gt_m=a, gt_to_pred_m=b)
