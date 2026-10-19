"""Selective exact GPU-index CD and unchanged project GPU IoU."""
from fpsgen.ops.gpu_kdtree import load_extension,chamfer_kdtree
from fpsgen.utils.selected_point_metrics import select_metrics,SelectedPointMetrics


class GPUKDTreeMetrics:
    def __init__(self,metrics=('cd','iou05'),*,geometry=None):
        self.metrics=select_metrics(metrics)
        self.extension=load_extension('/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/gpu_kdtree_v1') if 'cd' in self.metrics else None
        self.iou=SelectedPointMetrics(('iou05',),geometry=geometry) if 'iou05' in self.metrics else None

    def evaluate(self,predicted,target):
        value=chamfer_kdtree(predicted,target,extension=self.extension) if self.extension is not None else {}
        if self.iou is not None:value.update(self.iou.evaluate(predicted,target))
        return value

    def evaluate_batch(self,predictions,targets):
        if not predictions or len(predictions)!=len(targets):
            raise ValueError('One natural-length target per scene required')
        return [self.evaluate(p,q) for p,q in zip(predictions,targets)]
