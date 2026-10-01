"""Exact low-risk optimization checks for Legacy BEVFlow speed options."""
import torch

from fpsgen.models.gen_img import BEVDataProcessor
from fpsgen.datasets.dataloader.semantic_kitti import point_set_to_sparse


def _historical_layout(processor, pcd, labels):
    b, grid = pcd.shape[0], processor.grid_size
    pixels = (((pcd[:, :, :2].float() + processor.pc_range) / (processor.pc_range * 2) * grid)
              .long().clamp(0, grid - 1))
    flat = pixels[:, :, 0] * grid + pixels[:, :, 1]
    labels = labels.squeeze(-1)
    output = []
    for ids in ([10, 13, 18, 20, 252, 256, 258, 259], [40, 44, 48, 49]):
        found = torch.isin(labels, torch.tensor(ids, dtype=labels.dtype))
        values = torch.zeros((b, grid * grid), dtype=torch.float32)
        values.scatter_add_(1, flat, found.float())
        output.append((values > 0).float().view(b, 1, grid, grid))
    return torch.cat(output, dim=1)


def test_semantic_lut_matches_historical_layout_exactly():
    torch.manual_seed(3)
    processor = BEVDataProcessor(grid_size=16)
    points = torch.randn(3, 121, 3) * 80
    labels = torch.randint(-10, 320, (3, 121, 1))
    assert torch.equal(processor.get_layout_bev(points, labels), _historical_layout(processor, points, labels))


def test_zero_copy_sample_conversion_preserves_tensor_values():
    import numpy as np
    full = np.arange(24, dtype=np.float32).reshape(8, 3)
    part = np.arange(12, dtype=np.float32).reshape(4, 3)
    pose = np.eye(4, dtype=np.float32)
    labels_full = np.arange(8, dtype=np.float32).reshape(8, 1)
    labels_part = np.arange(4, dtype=np.float32).reshape(4, 1)
    legacy = point_set_to_sparse(full, part, "frame", pose, labels_part, labels_full, zero_copy_numpy=False)
    fast = point_set_to_sparse(full, part, "frame", pose, labels_part, labels_full, zero_copy_numpy=True)
    for old, new in zip(legacy, fast):
        if isinstance(old, torch.Tensor): assert torch.equal(old, new)
