import torch

from fpsgen.utils.bev_eval_metrics import (
    compute_bev_metrics,
    make_valid_disk_mask,
    process_generated_bev,
)


def _target(size=8):
    bev = torch.full((1, 3, size, size), -1.0)
    bev[:, 0, 2, 2] = .2
    bev[:, 1, 2, 2] = .3
    bev[:, 2, 2, 2] = 1.0
    return bev


def _metrics(pred, gt, inp=None):
    if inp is None:
        inp = torch.full_like(gt, -1.0)
    mask = torch.ones(gt.shape[-2:], dtype=torch.bool)
    return compute_bev_metrics(pred, gt, inp, mask)[0]


def test_perfect_prediction_is_perfect():
    gt = _target()
    out = _metrics(gt, gt)
    assert out['density_mass_tv'] == 0.0
    assert out['density_log_mae_union'] == 0.0
    assert out['height_mae_gtocc_m'] == 0.0
    assert out['occupancy_iou'] == 1.0
    assert out['occupancy_f1'] == 1.0


def test_density_error_does_not_change_height_or_occupancy():
    gt = _target()
    pred = gt.clone()
    pred[:, 0, 2, 2] = .8
    out = _metrics(pred, gt)
    assert out['density_log_mae_union'] > 0
    assert out['height_mae_gtocc_m'] == 0.0
    assert out['occupancy_iou'] == 1.0


def test_height_error_does_not_change_density_or_occupancy():
    gt = _target()
    pred = gt.clone()
    pred[:, 1, 2, 2] = .8
    out = _metrics(pred, gt)
    assert out['height_mae_gtocc_m'] > 0
    assert out['density_log_mae_union'] == 0.0
    assert out['occupancy_iou'] == 1.0


def test_occupancy_error_affects_end_to_end_not_intersection_value_metrics():
    gt = _target()
    pred = gt.clone()
    pred[:, 2, 2, 2] = 0.0  # strict production threshold: empty
    out = _metrics(pred, gt)
    assert out['occupancy_iou'] == 0.0
    assert out['height_mae_gtocc_m'] > 0
    assert out['density_log_mae_union'] > 0
    assert out['density_log_mae_intersection'] == 0.0


def test_strict_threshold_and_completion_region():
    gt = _target()
    pred = gt.clone()
    pred[:, 2, 2, 2] = 0.0
    assert process_generated_bev(pred)[0, 2, 2, 2] == -1
    inp = torch.full_like(gt, -1.0)
    recovered = gt.clone()
    out = _metrics(recovered, gt, inp)
    assert out['completion_recall'] == 1.0
    assert out['completion_f1'] == 1.0


def test_cells_outside_valid_disk_do_not_change_metrics():
    gt = _target(8)
    pred = gt.clone()
    disk = make_valid_disk_mask(8, 1.0)
    outside = (~disk).nonzero()[0]
    pred[:, :, outside[0], outside[1]] = torch.tensor([1.0, 1.0, 1.0])
    baseline = _metrics(gt, gt, torch.full_like(gt, -1.0))
    out = compute_bev_metrics(pred, gt, torch.full_like(gt, -1.0), disk)[0]
    for key in ('density_mass_tv', 'density_log_mae_union', 'height_mae_gtocc_m', 'occupancy_iou'):
        assert out[key] == baseline[key]
