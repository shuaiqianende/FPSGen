"""Small opt-in callbacks used by the long Stage-1 BEV training runs."""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


def _scalar_loss(outputs):
    """Return a detached scalar from Lightning's automatic-optimization output."""
    if isinstance(outputs, torch.Tensor):
        return float(outputs.detach().float().cpu())
    if isinstance(outputs, dict):
        value = outputs.get("loss")
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu())
    return float("nan")


def _finite_grads(module):
    return all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


class ThroughputCSVCallback(Callback):
    """Record synchronized step and loader-gap measurements at a small cadence."""

    fieldnames = [
        "global_step", "epoch", "loss", "optimizer_step_seconds",
        "data_ready_gap_seconds", "rolling_samples_per_second",
        "median_step_seconds", "p95_step_seconds", "median_data_gap_seconds",
        "p95_data_gap_seconds", "gpu_allocated_mb", "gpu_reserved_mb",
        "trailing_200_loss",
    ]

    def __init__(self, runtime_cfg):
        self.path = Path(runtime_cfg["throughput_csv"])
        self.summary_path = self.path.with_name(self.path.stem + "_loss_summary.csv")
        self.interval = int(runtime_cfg.get("throughput_interval_steps", 50))
        self.window = int(runtime_cfg.get("throughput_window_steps", 200))
        self.steps = deque(maxlen=self.window)
        self.gaps = deque(maxlen=self.window)
        self.losses = deque(maxlen=200)
        self.epoch_losses = []
        self._batch_started_at = None
        self._previous_batch_ended_at = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()
        with self.summary_path.open("w", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=["kind", "global_step", "epoch", "samples", "mean_loss"],
            ).writeheader()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_losses = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self._batch_started_at = now
        self._data_gap = (
            0.0 if self._previous_batch_ended_at is None
            else max(0.0, now - self._previous_batch_ended_at)
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero or self._batch_started_at is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ended_at = time.perf_counter()
        step_seconds = ended_at - self._batch_started_at
        self._previous_batch_ended_at = ended_at
        self.steps.append(step_seconds)
        self.gaps.append(self._data_gap)
        loss = _scalar_loss(outputs)
        self.losses.append(loss)
        self.epoch_losses.append(loss)
        # At this hook, Lightning has completed the optimizer step but has not
        # necessarily advanced ``global_step`` on every supported 1.x release.
        step = int(trainer.global_step) + 1
        if step % self.interval:
            return
        allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
        median_step = float(np.median(self.steps))
        row = {
            "global_step": step,
            "epoch": int(trainer.current_epoch),
            "loss": loss,
            "optimizer_step_seconds": step_seconds,
            "data_ready_gap_seconds": self._data_gap,
            "rolling_samples_per_second": len(batch["pcd_full"]) / median_step,
            "median_step_seconds": median_step,
            "p95_step_seconds": float(np.percentile(self.steps, 95)),
            "median_data_gap_seconds": float(np.median(self.gaps)),
            "p95_data_gap_seconds": float(np.percentile(self.gaps, 95)),
            "gpu_allocated_mb": allocated,
            "gpu_reserved_mb": reserved,
            "trailing_200_loss": float(np.nanmean(self.losses)),
        }
        with self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)
        if step % 500 == 0:
            self._write_summary("trailing_200", step, trainer.current_epoch, self.losses)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._write_summary(
                "epoch", int(trainer.global_step), trainer.current_epoch, self.epoch_losses
            )

    def _write_summary(self, kind, step, epoch, losses):
        finite = [value for value in losses if math.isfinite(value)]
        with self.summary_path.open("a", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=["kind", "global_step", "epoch", "samples", "mean_loss"],
            ).writerow({
                "kind": kind, "global_step": step, "epoch": int(epoch),
                "samples": len(finite),
                "mean_loss": float(np.mean(finite)) if finite else float("nan"),
            })


class FiniteTrainingPreflightCallback(Callback):
    """Fail a short preflight unless all losses/gradients stay finite and active."""

    def __init__(self, runtime_cfg):
        self.expected_steps = int(runtime_cfg.get("preflight_steps", 16))
        self.expected_batch_size = int(runtime_cfg.get("expected_batch_size", 8))
        self.steps_seen = 0
        self.condition_gradient_seen = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch_size = int(batch["pcd_full"].shape[0])
        if batch_size != self.expected_batch_size:
            raise RuntimeError(
                f"Preflight expected batch_size={self.expected_batch_size}, got {batch_size}"
            )

    def on_after_backward(self, trainer, pl_module):
        if not _finite_grads(pl_module):
            raise RuntimeError("Preflight found non-finite model gradients")
        condition_encoder = getattr(pl_module.model, "condition_encoder", None)
        if condition_encoder is None:
            raise RuntimeError("Preflight backbone does not expose condition_encoder")
        condition_grads = [
            parameter.grad for parameter in condition_encoder.parameters()
            if parameter.grad is not None
        ]
        if not all(torch.isfinite(grad).all().item() for grad in condition_grads):
            raise RuntimeError("Preflight found non-finite condition gradients")
        self.condition_gradient_seen |= any(grad.abs().sum().item() > 0 for grad in condition_grads)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = _scalar_loss(outputs)
        if not math.isfinite(loss):
            raise RuntimeError(f"Preflight found non-finite loss: {loss}")
        self.steps_seen += 1

    def on_fit_end(self, trainer, pl_module):
        if self.steps_seen != self.expected_steps:
            raise RuntimeError(
                f"Preflight expected {self.expected_steps} steps, completed {self.steps_seen}"
            )
        if not self.condition_gradient_seen:
            raise RuntimeError("Preflight did not observe a non-zero condition gradient")
        if trainer.is_global_zero:
            print(
                "PRELIGHT PASS: "
                f"steps={self.steps_seen}, batch_size={self.expected_batch_size}, "
                "losses_and_gradients=finite, condition_gradients=active"
            )
