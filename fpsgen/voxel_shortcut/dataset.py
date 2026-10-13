"""Strict sphere-aware GT; reuse audited numeric pairing and frozen split."""

import torch
from torch.utils.data import Dataset
from fpsgen.voxel_point.dataset import load_xyz, paired_frames, development_split, manifest_sha
from .geometry import ShortcutGeometry


def validate_inference_frames(rows):
    """The deployable input schema excludes training context before any model runs."""
    from pathlib import Path
    if not rows or any(set(row)!={'id','input'} for row in rows):
        raise ValueError('Inference manifest accepts id/input only')
    ids=[row['id'] for row in rows]
    if len(set(ids))!=len(ids) or any(Path(name).name!=name or name in ('.','..') for name in ids):
        raise ValueError('Inference IDs must be unique safe frame names')
    return rows


def validate_gt_points(gt,geometry,*,policy='strict',max_filter_distance_m=.001,max_filter_points=32):
    if gt.shape!=(geometry.point_budget,3):raise ValueError(f'Raw GT must be180000 XYZ, got{tuple(gt.shape)}')
    if not torch.isfinite(gt).all():raise ValueError('Nonfinite GT is never filtered')
    valid=geometry.valid_points(gt);removed=int((~valid).sum())
    if policy not in ('strict','filter_audited'):raise ValueError('Unknown GT validation policy')
    maximum_violation=0.
    if removed:
        p=gt[~valid].double()
        violation=torch.stack(((p.norm(dim=-1)-geometry.sphere_radius).clamp_min(0),
            (p.new_tensor(geometry.origin)-p).clamp_min(0).amax(-1),
            (p-p.new_tensor(geometry.maximum)).clamp_min(0).amax(-1)),1)
        maximum_violation=float(violation.max())
        if policy=='strict' or removed>max_filter_points or maximum_violation>max_filter_distance_m:
            raise ValueError(f'{removed} GT points outside ROI;max violation{maximum_violation}m;policy{policy}')
        gt=gt[valid]
    return gt,{'policy':policy,'raw_points':geometry.point_budget,'retained_points':len(gt),
        'filtered_points':removed,'max_violation_m':maximum_violation,'gt_mass_repaired':False,'crop':'3D norm<50 + grid'}


class ShortcutDataset(Dataset):
    def __init__(self, frames, geometry=None, **validation):
        self.frames = list(frames)
        self.geometry = geometry or ShortcutGeometry()
        self.validation=validation

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        try:
            xyz, gt = load_xyz(frame['input']), load_xyz(frame['gt'])
            gt,validation=validate_gt_points(gt,self.geometry,**self.validation)
            counts = self.geometry.counts(gt,expected_points=len(gt))
        except (ValueError,OSError) as exc:
            raise ValueError(f"Rejected{frame['sequence']}/{frame['frame']:06d}: {exc}") from exc
        return {'input_xyz':xyz,'gt_xyz':gt,'gt_counts':self.geometry.to_field(counts)[None],
                'observed':self.geometry.observed_mask(xyz)[0], 'frame':frame,
                'gt_validation':validation}
