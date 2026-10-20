"""Per-scene, inference-only PTv3 geometry cache.

Only immutable input geometry and its serialization are cached. A fresh Point
container and fresh feature tensor are built for every flow time step.
"""
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PTv3InferenceTopology:
    grid_coord: torch.Tensor
    batch: torch.Tensor
    offset: torch.Tensor
    coord: torch.Tensor
    sparse_shape: tuple
    serialized_code: torch.Tensor
    serialized_order: torch.Tensor
    serialized_inverse: torch.Tensor
    serialized_depth: int
    validated_batch_size: int
    level: str = "serialization"
    attention: dict = field(default_factory=dict)
    pooling: dict = field(default_factory=dict)

    @classmethod
    @torch.no_grad()
    def build(cls, coords, centers, batch, sparse_shape, order, depth, *, scene_count, level):
        from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point

        if coords.ndim != 2 or coords.shape[1] != 3 or centers.shape != coords.shape:
            raise ValueError("PTv3 topology expects matching [M,3] coordinates")
        if batch.shape != (len(coords),) or not len(coords):
            raise ValueError("PTv3 topology requires nonempty [M] batch")
        # The caller has already validated sorted contiguous batch IDs. In the
        # current inference contract this is a single scene; derive offset
        # without invoking Point.batch2offset's CUDA bincount.
        offsets = torch.bincount(batch.long(), minlength=scene_count).cumsum(0)
        scratch = Point({
            "feat": torch.zeros((len(coords), 1), device=coords.device),
            "coord": centers,
            "grid_coord": coords.int(),
            "batch": batch.long(),
            "offset": offsets,
            "sparse_shape": list(sparse_shape),
        })
        scratch.serialization(order=order, depth=depth, shuffle_orders=False)
        return cls(
            grid_coord=coords.int(), batch=batch.long(), offset=offsets,
            coord=centers, sparse_shape=tuple(sparse_shape),
            serialized_code=scratch.serialized_code,
            serialized_order=scratch.serialized_order,
            serialized_inverse=scratch.serialized_inverse,
            serialized_depth=int(scratch.serialized_depth),
            validated_batch_size=scene_count,
            level=level,
        )

    def make_point(self, features, context=None, *, batch1_fastpath=False):
        from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point

        data = {
            "feat": features,
            "coord": self.coord,
            "grid_coord": self.grid_coord,
            "batch": self.batch,
            "offset": self.offset,
            "sparse_shape": list(self.sparse_shape),
            "serialized_code": self.serialized_code,
            "serialized_order": self.serialized_order,
            "serialized_inverse": self.serialized_inverse,
            "serialized_depth": self.serialized_depth,
            "_v3_topology": self,
        }
        if batch1_fastpath and self.validated_batch_size == 1:
            data["_validated_batch_size"] = 1
        if context is not None:
            data["context"] = context
        return Point(data)
