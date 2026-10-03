"""Opt-in formal-run provenance and low-frequency production diagnostics.

No CUDA synchronization per batch. Each rank keeps a manifest so memory,
batch size and scaler outcomes are not inferred from rank zero alone.
"""
import csv
import datetime
import json
import os
import subprocess
import time
import statistics
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from fpsgen.utils.formal_bev import parameter_counts, source_fingerprint
from fpsgen.utils.training_callbacks import _loss_tensor


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class SustainedStallMonitor:
    """CPU-only 100-step windows; isolated bursts cannot stop production.

    First three steady windows calibrate wall time, excluding compile warmup.
    The reference is frozen, so a long slowdown cannot normalize itself away.
    Host gaps include framework/transfer/logging, not just storage I/O.
    """
    def __init__(self):
        self.calibration = []
        self.reference = None
        self.bad_windows = 0

    def observe(self, gaps, wall_seconds):
        stalls = sum(gap > 2 for gap in gaps)
        gap_seconds = sum(gaps)
        slow = self.reference is not None and wall_seconds > 1.5*self.reference
        severe = stalls >= .2*len(gaps) or gap_seconds >= 100
        bad = bool(severe or slow)
        self.bad_windows = self.bad_windows+1 if bad else 0
        if self.reference is None and not severe:
            self.calibration.append(wall_seconds)
            if len(self.calibration) == 3:
                self.reference = statistics.median(self.calibration)
        return dict(wall_seconds=wall_seconds, gap_count=len(gaps),
                    multi_second_gaps=stalls, total_host_gap_seconds=gap_seconds,
                    max_host_gap_seconds=max(gaps, default=0),
                    reference_window_seconds=self.reference,
                    consecutive_bad_windows=self.bad_windows,
                    warning=bool(stalls or bad), stop=self.bad_windows >= 3)


class FormalBEVRunCallback(Callback):
    def __init__(self, runtime):
        self.directory = Path(runtime["formal_manifest_dir"])
        self.strict_stalls = bool(runtime.get("formal_stall_gate", False))
        self.stall_policy = runtime.get("formal_stall_policy", "strict")
        if self.stall_policy not in ("strict", "sustained"):
            raise ValueError("Unknown formal stall policy")
        self.stall_monitor = SustainedStallMonitor()
        self.window_gaps = []
        self.window_started = None
        self.max_skip_fraction = float(runtime.get("formal_max_skip_fraction", .01))
        self.start = None
        self.losses = []
        self.samples = 0
        self.skips = 0
        self.steps = 0
        self.inf_flags = []
        self.gaps = []
        self.last_end = None
        self.batch_sizes = set()
        self.state = {}

    def write(self, trainer):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"run_manifest_rank{trainer.global_rank}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, default=str)+"\n")
        temporary.replace(path)
        if trainer.is_global_zero:
            # Rank-zero alias; rank files remain authoritative for each GPU.
            alias = self.directory / "run_manifest.json"
            temporary = alias.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.state, indent=2, default=str)+"\n")
            temporary.replace(alias)

    def on_fit_start(self, trainer, pl_module):
        cfg = dict(pl_module.hparams)
        if trainer.world_size != int(cfg["train"]["n_gpus"]):
            raise RuntimeError("Formal run GPU count differs from config; refusing silent single-GPU fallback")
        self.start = time.perf_counter()
        self.scaler = getattr(trainer.precision_plugin, "scaler", None)
        self.scale = self.scaler.get_scale() if self.scaler else None
        loader = trainer.train_dataloader
        loader = getattr(loader, "loaders", loader)
        if isinstance(loader, (list, tuple)) and len(loader) == 1:
            loader = loader[0]
        self.state = {
            "status": "RUNNING", "start_time": utc_now(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "local_source_fingerprint": source_fingerprint(Path(__file__).resolve().parents[2]),
            "model": cfg.get("model", {}).get("backbone", "legacy"), "config": cfg,
            "params": parameter_counts(pl_module.model),
            "gpu_model": torch.cuda.get_device_name(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "rank": trainer.global_rank, "world_size": trainer.world_size,
            "local_rank":trainer.local_rank,
            "cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES'),
            "per_gpu_batch": int(cfg["train"]["batch_size"]),
            "global_batch": int(cfg["train"]["batch_size"])*trainer.world_size,
            "train_frames": len(loader.dataset),
            "steps_per_epoch": int(trainer.num_training_batches),
            "planned_optimizer_steps": int(trainer.estimated_stepping_batches),
            "optimizer": {"name": "AdamW", "lr": cfg["train"]["lr"], "betas": [.9, .999],
                          "weight_decay": .0001, "warmup_steps": 1000, "schedule": "cosine"},
            "condition_states": [f"{i:03b}" for i in range(8)],
            "flow_rng_policy":"historical independent stream seeds shared across ranks; cross-rank noise/time/masks correlated",
            "geometry_precision": "fp32", "loss_precision": "fp32",
            "compile_requested": cfg.get("runtime", {}).get("compile_dense", False),
            "activation_checkpointing": cfg.get("runtime", {}).get("activation_checkpointing", False),
            "grad_scaler_initial_scale": self.scale,
            "compile_startup_included_in_first_interval": True,
            "skipped_optimizer_steps": 0,
            "gap_measurement": "host batch-end to next batch-start, NOT pure DataLoader or disk wait",
            "stall_policy": self.stall_policy,
        }
        self.write(trainer)

    def on_train_epoch_start(self, trainer, pl_module):
        self.losses = []
        self.epoch_started = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        actual = int(batch["pcd_full"].shape[0])
        if actual > self.state["per_gpu_batch"]:
            raise RuntimeError("Unexpected batch larger than per-GPU contract")
        self.batch_sizes.add(actual)
        self.samples += actual
        now = time.perf_counter()
        if self.last_end is not None and self.steps >= 100:
            self.gaps.append(now-self.last_end)
            self.window_gaps.append(now-self.last_end)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
        # Lightning1.8 calls this AFTER GradScaler.unscale_. Preserve the
        # actual found_inf tensor before scaler.update clears its state.
        # No per-step item(), get_scale() or CUDA synchronization is needed.
        if self.scaler:
            flags = self.scaler._found_inf_per_device(optimizer)
            if not flags:
                raise RuntimeError("GradScaler overflow flags unavailable at unscaled optimizer hook")
            self.inf_flags.append(torch.stack(tuple(flags.values())).sum().detach())

    def collect_skips(self):
        if self.inf_flags:
            self.skips += int((torch.stack(self.inf_flags)>0).sum().cpu())
            self.inf_flags.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = _loss_tensor(outputs)
        if loss is not None:
            self.losses.append(loss)
        self.steps += 1
        self.last_end = time.perf_counter()
        if self.steps % 100 == 0:
            self.collect_skips()
            self.scale = self.scaler.get_scale() if self.scaler else None
            mean = torch.stack(self.losses[-100:]).mean()
            if not torch.isfinite(mean).item():
                raise FloatingPointError("Non-finite formal training loss")
            if self.steps >= 500 and self.skips/self.steps > self.max_skip_fraction:
                raise FloatingPointError("FP16 overflow skip fraction exceeds recorded formal gate")
            self.state.update({"epoch": int(trainer.current_epoch), "optimizer_steps": int(trainer.global_step),
                "observed_batches": self.steps, "skipped_optimizer_steps": self.skips,
                "grad_scaler_scale": self.scale,
                "peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_mb": torch.cuda.max_memory_reserved()/2**20,
                "rank_seen_samples": self.samples, "wall_seconds": time.perf_counter()-self.start})
            self.state['total_seen_samples']=self.samples*trainer.world_size
            now = time.perf_counter()
            if self.strict_stalls and self.stall_policy == "sustained":
                if self.window_started is not None and self.window_gaps:
                    evidence = self.stall_monitor.observe(self.window_gaps, now-self.window_started)
                    self.state['last_stall_window'] = evidence
                    path = self.directory/f"stall_windows_rank{trainer.global_rank}.csv"
                    exists = path.exists()
                    with path.open('a', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=['step', *evidence])
                        if not exists: writer.writeheader()
                        writer.writerow(dict(step=self.steps, **evidence))
                    if evidence['warning']:
                        print(f"STALL WARNING rank={trainer.global_rank} step={self.steps}: {evidence}", flush=True)
                    if evidence['stop']:
                        self.state.update(status='FAILED', end_time=utc_now(),
                            exception='Sustained slowdown across three consecutive100-step windows')
                        self.write(trainer)
                        raise RuntimeError(self.state['exception'])
                self.window_started = now
                self.window_gaps.clear()
            self.write(trainer)
            if self.strict_stalls and self.stall_policy == "strict" and len(self.gaps) >= 100:
                stalls = sum(gap > 2 for gap in self.gaps)
                if stalls/len(self.gaps) > .01:
                    raise RuntimeError("Repeated multi-second host-ready stalls; STOP formal queue")

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.losses:
            return
        mean = torch.stack(self.losses).mean()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean)
            mean /= trainer.world_size
        value = float(mean.cpu())
        if trainer.is_global_zero:
            path = self.directory/"epoch_loss.csv"
            exists = path.exists()
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("epoch", "global_step", "mean_loss", "wall_seconds"))
                if not exists:
                    writer.writeheader()
                writer.writerow({"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                                 "mean_loss": value, "wall_seconds": time.perf_counter()-self.epoch_started})
            print(f"FORMAL EPOCH {trainer.current_epoch+1}: global_step={trainer.global_step} mean_loss={value:.6f}", flush=True)

    def on_fit_end(self, trainer, pl_module):
        # Save final even when the 10-epoch pilot is shorter than cadence50.
        # All ranks must enter save_checkpoint; only rank zero writes in DDP.
        trainer.save_checkpoint(str(self.directory/"final.ckpt"))
        self.collect_skips()
        self.scale = self.scaler.get_scale() if self.scaler else None
        from torch._dynamo.utils import counters
        import numpy as np
        self.state.update({"status": "COMPLETE", "end_time": utc_now(),
            "completed_epochs": int(trainer.current_epoch), "optimizer_steps": int(trainer.global_step),
            "wall_seconds": time.perf_counter()-self.start, "rank_seen_samples": self.samples,
            "total_seen_samples":self.samples*trainer.world_size,
            "skipped_optimizer_steps": self.skips, "grad_scaler_scale": self.scale,
            "actual_optimizer_updates": self.steps-self.skips,
            "observed_local_batch_sizes": sorted(self.batch_sizes),
            "dynamo_stats": {name:dict(counters[name]) for name in ('stats','frames','graph_break')},
            "peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved()/2**20,
            "host_ready_gap_ms": {key: float(v) for key,v in zip(("p50", "p95", "p99", "max"),
                 np.percentile(np.asarray(self.gaps)*1000, (50,95,99,100)))} if self.gaps else None})
        self.write(trainer)

    def on_exception(self, trainer, pl_module, exception):
        self.state.update(status="FAILED", exception=f"{type(exception).__name__}: {exception}", end_time=utc_now())
        self.write(trainer)
