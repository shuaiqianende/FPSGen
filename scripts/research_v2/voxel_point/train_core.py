"""Bounded smoke and gate-controlled B1 training, sharing successful-update semantics."""

import contextlib
import gc
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from common import (parser, read_config, frames_for, configure_runtime, save_json,
                    load_checkpoint, require_formal_gate, manifest_sha)
from fpsgen.voxel_point.dataset import VoxelPointDataset, DistributedSceneBatchSampler, collate_scenes
from fpsgen.voxel_point.anchor_sampler import sample_gaussian_anchors
from fpsgen.voxel_point.density_loss import density_loss


def sample_source(sample, cfg, spec, generator, device, *, use_predicted, formal=False):
    if use_predicted:
        cache = cfg.get("predicted_counts_cache")
        if not cache:
            raise ValueError("Predicted source selected but no count cache configured")
        from pathlib import Path
        frame = sample["frame"]
        path = Path(cache) / f"{frame['sequence']}_{frame['frame']:06d}.npz"
        with np.load(path, allow_pickle=False) as value:
            raw_counts = value["counts"]
            if raw_counts.dtype.kind not in "iu":
                raise ValueError("Predicted count cache must contain integer counts")
            counts = torch.from_numpy(raw_counts.copy()).long().to(device)
            metadata = json.loads(str(value["metadata"].item()))
        if metadata["geometry"] != spec.to_dict() or metadata.get("point_count") != 180000:
            raise ValueError("Predicted cache geometry/count mismatch")
        if metadata.get("source") != "predicted_density" or not metadata.get("stage1_checkpoint_sha256"):
            raise ValueError("Missing predicted-source provenance")
        if formal and metadata.get("mechanics_smoke_only", True):
            raise ValueError("Formal training cannot consume smoke predicted-source caches")
        if (metadata["frame"]["sequence"], metadata["frame"]["frame"]) != (frame["sequence"],frame["frame"]):
            raise ValueError("Predicted count cache frame pairing mismatch")
    else:
        counts = spec.dense_to_flat(sample["gt_counts"]).to(device)
    if counts.shape != (spec.num_voxels,) or int(counts.sum()) != 180000:
        raise ValueError("Source cache must allocate exactly 180000 points")
    anchors, ids = sample_gaussian_anchors(spec, counts, generator=generator)
    return anchors, ids


def main(expected_stage):
    args = parser(__doc__, training=True).parse_args()
    cfg, spec, sha = read_config(args.config)
    if cfg["stage"] != expected_stage:
        raise ValueError("Script/config stage mismatch")
    require_formal_gate(args, expected_stage)
    if args.max_frames < 0 or args.max_steps < 0:
        raise ValueError("Limits cannot be negative")
    device, smi = configure_runtime()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_num_threads(4)
    frames = frames_for(cfg, "train", args.max_frames or (1 if args.smoke else 0))
    dataset = VoxelPointDataset(frames, spec, gt_roi_policy=cfg["data"].get("gt_roi_policy", "strict"))
    if not frames:
        raise ValueError("Empty training manifest")
    if expected_stage == "stage1":
        from fpsgen.voxel_point.voxdet_density import VoxDetDensity
        model = VoxDetDensity(spec, activation_checkpointing=cfg["runtime"]["activation_checkpointing"], **cfg["model"])
    elif expected_stage == "teacher":
        from fpsgen.voxel_point.mapping_teacher import MappingTeacher
        model = MappingTeacher(spec, cfg["network"])
    else:
        from fpsgen.voxel_point.mapping_teacher import MappingTeacher
        from fpsgen.voxel_point.point_flow import PointFlow
        model = PointFlow(spec, cfg["network"])
        teacher = MappingTeacher(spec, cfg["network"]).to(device).eval()
        load_checkpoint(teacher, cfg.get("teacher_checkpoint"), spec, stage="teacher")
        teacher.requires_grad_(False)
        if not args.smoke:
            if not cfg.get("pseudo_target_gate"):
                raise ValueError("Formal Student requires an accepted pseudo-target gate report")
            gate = json.loads(open(cfg["pseudo_target_gate"]).read())
            if not gate.get("pseudo_targets_reliable"):
                raise ValueError("Student pseudo-target quality gate failed")
    model.to(device).train()
    parameters = sum(p.numel() for p in model.parameters())
    raw_model = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    train = cfg["train"]
    batch_per_gpu = cfg["runtime"].get("batch_per_gpu", 1)
    if batch_per_gpu < 1 or (expected_stage != "stage1" and batch_per_gpu != 1):
        raise ValueError("Batch must be positive; point stages retain B1 and 180k points")
    accumulation = train["gradient_accumulation"]
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    global_batch = world * batch_per_gpu * accumulation
    budget = math.ceil(train["target_seen_frames"] / global_batch) if expected_stage == "stage1" else math.ceil(train["epochs"] * len(frames) / global_batch)
    steps = args.max_steps or (train["smoke_steps"] if args.smoke else budget)
    num_workers = cfg["runtime"].get("data_workers", 0) if expected_stage == "stage1" else 0
    if num_workers < 0:
        raise ValueError("data_workers cannot be negative")
    sampler = DistributedSceneBatchSampler(len(dataset), batch_per_gpu, rank, world,
        (max(64, steps*4)+1)*accumulation, shuffle=train.get("shuffle", False), seed=cfg["seed"])
    loader_options = {"num_workers":num_workers,"pin_memory":num_workers > 0}
    if num_workers:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    loader = torch.utils.data.DataLoader(dataset,batch_sampler=sampler,collate_fn=collate_scenes,**loader_options)
    scene_batches = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    scheduler = None
    if expected_stage == "stage1":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=train["lr"],total_steps=max(32,steps),pct_start=train["pct_start"],cycle_momentum=False)
    amp = args.precision == "fp16" if args.precision else cfg["runtime"]["amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + rank)
    source_generator = torch.Generator().manual_seed(cfg["seed"] + 10000 + rank)
    source_schedule = []
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / f"preflight_rank{rank}.json", {"nvidia_smi":smi,"physical_visible":os.environ["CUDA_VISIBLE_DEVICES"],"config_sha256":sha,"parameter_count":parameters,"amp":amp,"world_size":world,"batch_per_gpu":batch_per_gpu,"global_batch":global_batch,"frames_sha256":manifest_sha(frames),"gt_roi_policy":dataset.gt_roi_policy})
    milestones = {max(1,math.ceil(f*steps)) for f in train.get("checkpoint_fractions", (.25,.5,.75,1.))}
    checkpoint_every = train.get("checkpoint_every_steps", 0)
    checkpoint_initial = train.get("checkpoint_initial_steps", 0)
    successful, attempts, exposed = 0, 0, 0
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    while successful < steps:
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        detail = []
        for micro in range(accumulation):
            first_index = ((attempts * accumulation + micro) * world + rank) * batch_per_gpu
            samples = next(scene_batches)
            sample = samples[0]
            epoch = first_index / len(dataset)
            gt = sample["gt_xyz"].to(device) if expected_stage != "stage1" else None
            exposed += len(samples)
            source_kind = "supervised_density"
            if expected_stage != "stage1":
                warmup = expected_stage == "teacher" and epoch < train.get("warmup_epochs",0)
                fraction = train["predicted_source_fraction"]
                if fraction == .5:
                    if not source_schedule and not warmup:
                        source_schedule = [bool(value) for value in torch.randperm(2,generator=source_generator)]
                    predicted = False if warmup else source_schedule.pop()
                else:
                    predicted = not warmup and bool(torch.rand((),generator=source_generator) < fraction)
                # Smoke mechanics use GT anchors; this is explicitly recorded.
                if args.smoke and not cfg.get("predicted_counts_cache"):
                    predicted = False
                source_kind = "predicted_density" if predicted else "gt_density"
                anchors, ids = sample_source(sample,cfg,spec,generator,device,use_predicted=predicted,formal=not args.smoke)
            sync_context = model.no_sync() if world > 1 and micro+1 < accumulation else contextlib.nullcontext()
            with sync_context:
                with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                    if expected_stage == "stage1":
                        inputs = [s["input_xyz"].to(device,non_blocking=True) for s in samples]
                        outputs = model(inputs)
                        batched_counts = torch.stack([s["gt_counts"].to(device,non_blocking=True) for s in samples])[:,None]
                        terms = density_loss(outputs, batched_counts, spec, cfg["loss_weights"])
                        loss = terms["total"]
                        values = {k:float(v.detach()) for k,v in terms.items()}
                    elif expected_stage == "teacher":
                        from fpsgen.voxel_point.metrics import chamfer_sqrt
                        endpoint = model(anchors,ids,gt)
                        loss = chamfer_sqrt(endpoint,gt,backend=cfg["runtime"]["knn_backend"])
                        values = {"teacher_cd":float(loss.detach())}
                    else:
                        with torch.no_grad():
                            endpoint = teacher(anchors,ids,gt)
                            if cfg.get("sinkhorn"):
                                from fpsgen.voxel_point.sinkhorn_refiner import refine_with_policy
                                endpoint, diagnostics = refine_with_policy(endpoint,gt,spec,backend=cfg["runtime"]["knn_backend"],reset_peak_memory=False,**cfg["sinkhorn"])
                            else:
                                diagnostics = {"fallback_used":True}
                        from fpsgen.voxel_point.point_flow import flow_matching_pair
                        t = torch.rand((1,),device=device,generator=generator)
                        state, velocity = flow_matching_pair(anchors,endpoint,t)
                        prediction = model(state,t,ids)
                        loss = torch.nn.functional.mse_loss(prediction.float(),velocity.float())
                        values = {"velocity_mse":float(loss.detach()),"refined_target":not diagnostics["fallback_used"]}
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss / accumulation).backward()
            for scene in samples:
                detail.append({"frame":scene["frame"],"gt_validation":scene["gt_validation"],
                               "source_kind":source_kind,"loss_scope":"batch",**values})
        scaler.unscale_(optimizer)
        gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not gradients_finite and not amp:
            raise FloatingPointError("Nonfinite FP32 gradients")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= previous_scale
        attempts += 1
        if stepped:
            successful += 1
            if scheduler:
                scheduler.step()
        if attempts > max(64,steps*4):
            raise FloatingPointError("Too many failed AMP updates")
        last_loss = float(loss.detach())
        if expected_stage == "stage1":
            del outputs, terms, loss, gt, inputs, batched_counts
        elif expected_stage == "teacher":
            del endpoint, loss, anchors, ids, gt
        else:
            del prediction, loss, endpoint, state, velocity, anchors, ids, gt
        # PTv3 hierarchy + checkpoint closures can retain Point parent cycles.
        # Collect after removing scene graphs; do not accumulate prior scenes.
        if expected_stage != "stage1":
            gc.collect()
        torch.cuda.synchronize(device)
        row = {"successful_updates":successful,"attempts":attempts,"gradients_finite":gradients_finite,
            "step_seconds":time.perf_counter()-step_start,"peak_vram_bytes":torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"details":detail,
            "global_batch":global_batch,"learning_rate":optimizer.param_groups[0]["lr"]}
        rows.append(row)
        with (args.output/f"training_rank{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
        print(f"rank={rank} update={successful}/{steps} loss={last_loss:.5f} sec={row['step_seconds']:.2f}",flush=True)
        periodic_checkpoint = checkpoint_every > 0 and successful % checkpoint_every == 0
        initial_checkpoint = checkpoint_initial > 0 and successful == checkpoint_initial
        if stepped and (successful in milestones or periodic_checkpoint or initial_checkpoint) and rank == 0:
            checkpoint_path = args.output/f"checkpoint_{successful:06d}.pt"
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"stage":expected_stage,"geometry":spec.to_dict(),"config_sha256":sha,"config":cfg,
                "model":raw_model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                "scheduler":scheduler.state_dict() if scheduler else None,"successful_updates":successful,
                "seen_frames":successful*global_batch,"attempted_frame_exposures":exposed*world,
                "mechanics_smoke_only":args.smoke},temporary_path)
            temporary_path.replace(checkpoint_path)
    elapsed = time.perf_counter()-start
    source_counts = {}
    refined_count = 0
    for row in rows:
        for value in row["details"]:
            name = value["source_kind"]
            source_counts[name] = source_counts.get(name,0)+1
            refined_count += int(value.get("refined_target",False))
    summary = {"stage":expected_stage,"status":"PASS" if all(row["gradients_finite"] for row in rows) else "FAIL_NONFINITE_GRADIENT",
        "mechanics_smoke_only":args.smoke,
        "source_distribution_validated":not args.smoke,"config_sha256":sha,"parameter_count":parameters,
        "successful_updates":successful,"attempts":attempts,"seen_frames":successful*global_batch,
        "attempted_frame_exposures":exposed*world,
        "all_gradients_finite":all(row["gradients_finite"] for row in rows),
        "source_counts":source_counts,"refined_target_count":refined_count,
        "peak_vram_bytes":max(row["peak_vram_bytes"] for row in rows),
        "peak_reserved_bytes":max(row["peak_reserved_bytes"] for row in rows),"seconds":elapsed,"mean_step_seconds":elapsed/attempts,
        "physical_visible":os.environ["CUDA_VISIBLE_DEVICES"],"world_size":world,
        "batch_per_gpu":batch_per_gpu,"global_batch":global_batch,
        "gt_roi_policy":dataset.gt_roi_policy,
        "filtered_roi_points":sum(v["gt_validation"]["filtered_roi_points"] for row in rows for v in row["details"])}
    save_json(args.output/f"summary_rank{rank}.json",summary)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
