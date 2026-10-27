"""Registered noise0 -> data1 MeanFlow-V2 objectives; legacy MF is untouched.

AW epsilon has units of per-valid-voxel scene MSE, not spatially summed error.
Only the prediction carries reverse AD. Native forward AD holds s fixed.
"""
from contextlib import contextmanager
import torch
from .meanflow_jvp import endpoint_jvp, uncached_current_autocast, without_activation_checkpointing
from .meanflow_math import meanflow_loss, broadcast_scene, scene_squared_errors

ARMS = ('MFV2-A', 'MFV2-B')
EPSILON = 0.01
GAP_EDGES = (0., .125, .25, .5, .75, 1.)


@contextmanager
def preserved_target_buffers(model):
    """Extra detached target passes must not advance normalization state.

    Restore in finally, including exception paths; do not toggle training mode
    (which would change the target function). The real prediction is outside.
    """
    buffers = [(b, b.detach().clone()) for b in model.buffers()] if model is not None else []
    flags = [(m, m.training) for m in model.modules()] if model is not None else []
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                if not torch.equal(buffer, original):
                    buffer.copy_(original)
        if any(m.training != flag for m, flag in flags):
            raise RuntimeError('Target forward changed module training flags')


def adaptive_scene_objective(per_scene):
    if per_scene.ndim != 1 or not len(per_scene):
        raise ValueError('One valid-voxel FP32 MSE per scene is required')
    if not bool(torch.isfinite(per_scene).all()) or bool((per_scene < 0).any()):
        raise FloatingPointError('Nonfinite/negative raw scene MSE')
    weights = (per_scene.detach() + EPSILON).pow(-1.)
    return (weights * per_scene).mean(), weights


def meanflow_v2_loss(u_fn, state, time, time_end, velocity, valid, *, arm,
                     model=None, skip_all_diagonal=True):
    if arm not in ARMS:
        raise ValueError('Explicit MFV2-A or MFV2-B arm required')
    if state.shape != velocity.shape or time.shape != time_end.shape or time.ndim != 1:
        raise ValueError('Dense scene/state/time shapes differ')
    if not bool((torch.isfinite(time) & torch.isfinite(time_end) & (time >= 0)
                 & (time_end >= time) & (time_end <= 1)).all()):
        raise ValueError('Require forward 0<=t<=s<=1')
    mask = valid.bool().expand_as(state)
    state = torch.where(mask, state, 0.)
    velocity = torch.where(mask, velocity, 0.)
    delta = time_end - time
    duration = broadcast_scene(delta, state)
    boundary = None
    if arm == 'MFV2-A':
        # Exact legacy JVP, sign, detached u-target, and 75% diagonal shortcut.
        _, detail = meanflow_loss(u_fn, state, time, time_end, velocity, mask,
                                  model=model, adaptive_power=1., adaptive_epsilon=EPSILON,
                                  skip_all_diagonal=skip_all_diagonal)
        prediction, target = detail['prediction'], detail['target']
        derivative = detail['derivative']
        per_scene = detail['per_scene']
        boundary_calls = 0
    else:
        if skip_all_diagonal and bool((delta == 0).all()):
            derivative = torch.zeros_like(velocity)
            jvp_calls = boundary_calls = 0
        else:
            from contextlib import nullcontext
            checkpoint = without_activation_checkpointing(model) if model is not None else nullcontext()
            with preserved_target_buffers(model), checkpoint, torch.no_grad(), uncached_current_autocast(state.device.type):
                boundary = torch.where(mask, u_fn(state, time, time), 0.).detach()
            if not bool(torch.isfinite(boundary).all()):
                raise FloatingPointError('Nonfinite detached boundary velocity')
            with preserved_target_buffers(model):
                _, derivative = endpoint_jvp(u_fn, state, time, time_end, boundary, model=model)
            derivative = torch.where(mask, derivative, 0.).detach()
            jvp_calls = boundary_calls = 1
        if not bool(torch.isfinite(derivative).all()):
            raise FloatingPointError('Nonfinite native iMF derivative')
        u = torch.where(mask, u_fn(state, time, time_end), 0.)
        # Forward-time identity: v = u - (s-t) D_t u, with s fixed.
        prediction = torch.where(mask, u.float() - duration * derivative.float().detach(), 0.)
        target = velocity.detach()
        per_scene = scene_squared_errors(prediction, target, mask)
        detail = dict(prediction=prediction, target=target, derivative=derivative,
                      per_scene=per_scene, jvp_calls=jvp_calls, diagonal=delta == 0, duration=delta)
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise FloatingPointError('Nonfinite MeanFlow-V2 prediction/target')
    loss, weights = adaptive_scene_objective(per_scene)
    detail.update(weights=weights, boundary_velocity=boundary,
                  boundary_forwards=boundary_calls, velocity=velocity,
                  derivative_term=(duration * derivative.float()).detach(), time=time.detach(),
                  time_end=time_end.detach(), raw_unweighted_mse=per_scene.mean().detach())
    return loss, detail


@torch.no_grad()
def scene_diagnostics(detail, binary_target, valid):
    """Small CPU scalars only in logs; invalid voxels excluded independently."""
    mask = valid.bool().expand_as(binary_target)
    prediction, target = detail['prediction'].float(), detail['target'].float()
    error = (prediction - target).square()
    rows = []
    for i in range(len(binary_target)):
        chosen = mask[i]
        def rms(x):
            values = x[i].float()[chosen]
            return float(values.square().mean().sqrt())
        pos = chosen & (binary_target[i] > 0)
        neg = chosen & (binary_target[i] < 0)
        boundary = detail['boundary_velocity']
        row = dict(raw_mse=float(detail['per_scene'][i]), weight=float(detail['weights'][i]),
                   weighted_loss=float(detail['per_scene'][i] * detail['weights'][i]),
                   t=float(detail['time'][i]), s=float(detail['time_end'][i]),
                   delta=float(detail['duration'][i]), diagonal=bool(detail['diagonal'][i]),
                   jvp_rms=rms(detail['derivative']),
                   jvp_max_abs=float(detail['derivative'][i].float()[chosen].abs().max()),
                   conditional_velocity_rms=rms(detail['velocity']),
                   derivative_target_term_rms=rms(detail['derivative_term']),
                   v_est_rms=None if boundary is None else rms(boundary),
                   v_est_conditional_error_rms=None if boundary is None else rms(boundary.float() - detail['velocity']),
                   positive_mse=float(error[i][pos].mean()) if bool(pos.any()) else None,
                   negative_mse=float(error[i][neg].mean()) if bool(neg.any()) else None,
                   positive_count=int(pos.sum()), negative_count=int(neg.sum()),
                   valid_count=int(chosen.sum()), target_buffers_preserved=True)
        rows.append(row)
    return rows
