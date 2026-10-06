#!/usr/bin/env python3
"""Full held-out seq08, exact seed42 per frame, final Stage1 BEV checkpoints."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.evaluation.evaluate_completed_bev_study import (
    ROOT, BASE, OLD, DIP, LEGACY, gpu_free, read, rows, save, write_csv, final_losses,
)

METRICS = ("occupancy_iou", "occupancy_f1", "completion_f1", "height_mae_gtocc_m", "density_mass_tv")


def validate_full_result(directory, cfg, expected_frames=None, dense_backend='inductor'):
    summary = read(directory / "summary.json")
    expected = dict(condition="100", gt_dir="gt_possion", frame_stride=1,
                    generation_seeds=[42], samples_per_frame=1, bev_steps=10,
                    guidance_scale=cfg, runtime_precision="fp16", dense_backend=dense_backend,
                    checkpoint_global_step=299000, layout_source="literal_all_zero_no_semantic_layout")
    if any(summary.get(k) != v for k, v in expected.items()) or summary["checkpoint_epoch"] not in [499, 500]:
        raise ValueError(f"Fullseq08/seed42 evaluation mismatch: {directory}")
    frames = (directory / "frame_manifest.txt").read_text().splitlines()
    if len(frames) != 4071 or len(set(frames)) != 4071 or summary["num_frames"] != 4071:
        raise ValueError("Expected ALL4071 available seq08 input/GT frame pairs")
    if expected_frames is not None and frames != expected_frames:
        raise ValueError("Methods used different complete frame manifests")
    samples = rows(directory / "samples.csv")
    if [r["frame"] for r in samples] != frames or any(int(r["seed"]) != 42 for r in samples):
        raise ValueError("Wrong frame identities/order/noise seeds")
    values = np.asarray([[float(r[m]) for m in METRICS] for r in samples])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite full-sequence metrics")
    return summary, frames, values


def bootstrap(difference):
    rng = np.random.default_rng(20261001)
    results = []
    for _ in range(20):
        draws = rng.integers(len(difference), size=(500, len(difference)))
        results.append(difference[draws].mean(axis=1))
    boot = np.concatenate(results)
    return dict(mean_delta_fpsgen_minus_baseline=float(difference.mean()),
                ci95=np.quantile(boot, [.025, .975]).tolist(),
                bootstrap_resamples=10000, bootstrap_seed=20261001)


def archive_incomplete(root, model, cfg):
    """Preserve failed/partial artifacts before retry; never touch successes."""
    directory = root / model / f'cfg{cfg}'
    if (directory / 'summary.json').exists():
        raise ValueError('Completed result must be validated, not archived')
    stage = f'{model}_cfg{cfg}'
    log = root / (stage + '.log')
    if directory.exists() or log.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = root / 'failed_attempts' / (stage + '_' + stamp)
        archive.mkdir(parents=True, exist_ok=False)
        if directory.exists():
            shutil.move(str(directory), str(archive / 'result'))
        if log.exists():
            shutil.move(str(log), str(archive / log.name))
        return str(archive)
    return None


def summarize(root, checkpoints, cfg1_backend='inductor'):
    quality, train, comparisons, data = [], [], {}, {}
    expected_frames = None
    input_reference = {}
    for model in checkpoints:
        hashes = []
        for cfg in [1, 2]:
            backend=cfg1_backend if cfg==1 else 'inductor'
            summary, frames, values = validate_full_result(root / model / f"cfg{cfg}", cfg, expected_frames,backend)
            expected_frames = frames
            if cfg in input_reference and summary['input_bev'] != input_reference[cfg]:
                raise ValueError('Input LiDAR BEV metrics differ between methods')
            input_reference[cfg] = summary['input_bev']
            hashes.append(summary['checkpoint_sha256'])
            data[(model,cfg)] = values
            row = dict(method=model, completed_epochs=500, optimizer_steps=299000, frames=len(frames),
                       generation_seed=42, cfg=cfg, euler_steps=10,
                       runtime_precision='fp16',dense_backend=backend,
                       runtime_fixes=summary.get('runtime_fixes',[]),
                       **{m:float(values[:,i].mean()) for i,m in enumerate(METRICS)},
                       **{m+'_frame_std':float(values[:,i].std()) for i,m in enumerate(METRICS)})
            quality.append(row)
        if len(set(hashes)) != 1:
            raise ValueError('CFG settings use different checkpoints')
        if model == 'FPSGen':
            reuse = read(OLD/'legacy/reuse_manifest.json');params=reuse['params']
            directory=ROOT/'outputs/research_v2/bevflow_formal_b16_gpu23_fp16_inductor'
            provenance=dict(execution='500ep DDP2 x batch16',wall_seconds=None,skipped_updates=None,
                            historical_limitations='No complete original rank manifests/scaler skip count')
        elif model == 'SiD2-style':
            directory=OLD/'sid2_s_formal';manifests=[read(directory/f'run_manifest_rank{i}.json') for i in [0,1]]
            params=manifests[0]['params']
            provenance=dict(execution='500ep DDP2 x batch16',wall_seconds=max(d['wall_seconds'] for d in manifests),
                            skipped_updates_by_rank=[d['skipped_optimizer_steps'] for d in manifests],
                            peak_allocated_mib=max(d['peak_allocated_mb'] for d in manifests))
        else:
            directory=DIP;manifest=read(DIP/'run_manifest_rank0.json');params=manifest['params']
            provenance=dict(execution='epoch1-50 DDP2 x16; epoch51-500 GPU3 x32',
                            continuation_wall_seconds=manifest['wall_seconds'],
                            continuation_skipped_updates=manifest['skipped_optimizer_steps'],
                            peak_allocated_mib_continuation=manifest['peak_allocated_mb'],
                            limitation='Global batch32 preserved; world-size/data ordering/private RNG stream changed at resume')
        train.append(dict(method=model,params_core=params['core'],params_condition=params['condition'],
                          params_pointpillar=params['pointpillar'],params_total=params['total'],
                          **final_losses(directory/'throughput_loss_summary.csv'),**provenance))
    for model in ['SiD2-style','DiP-S16']:
        comparisons[model]={}
        for cfg in [1,2]:
            differences=data[('FPSGen',cfg)]-data[(model,cfg)]
            comparisons[model][f'cfg{cfg}']={}
            for i,m in enumerate(METRICS):
                result=bootstrap(differences[:,i]);result['better_delta_sign']='negative' if i>=3 else 'positive'
                comparisons[model][f'cfg{cfg}'][m]=result
    write_csv(root/'final_quality_full_seq08_seed42.csv',quality)
    write_csv(root/'training_and_parameters.csv',train)
    save(root/'paired_bootstrap_full_seq08_seed42.json',dict(comparisons=comparisons,
         caveat='Frame-level paired descriptive uncertainty; temporal seq08 frames are correlated; one training/generation seed, not across-seed or architecture-wide significance'))
    save(root/'final_report.json',dict(quality=quality,training_and_parameters=train,paired=comparisons,
         scope='Final500epoch Stage1BEV; full4071 seq08 frames; seed42; FP16 dense,FP32 geometry/loss/metrics; not PointFlow'))
    print('FULL SEQ08 FINAL COMPARISON',json.dumps(quality),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',
                        help='Validate and reuse completed stages; archive failed attempts before retry')
    parser.add_argument('--inductor-beta0-fix',action='store_true',
                        help='Explicit process-local PyTorch2.0 beta=0 decomposition correction; no precision fallback')
    parser.add_argument('--cfg1-dense-backend',choices=['inductor','eager'],default='inductor',
                        help='Explicit documented CFG1 backend exception; FP16/sampling protocol unchanged')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':
        raise ValueError('Use physicalGPU3 only')
    root=args.output.resolve()
    if not root.is_relative_to(ROOT/'outputs'):
        raise ValueError('Use fresh repository output directory')
    gpu_free()
    if args.resume:
        if not (root/'campaign_manifest.json').is_file():
            raise FileNotFoundError('Resume requires an existing campaign manifest')
    else:
        root.mkdir(parents=True,exist_ok=False)
    checkpoints={'FPSGen':Path(read(OLD/'legacy/reuse_manifest.json')['checkpoint']),
                 'SiD2-style':OLD/'sid2_s_formal/final.ckpt','DiP-S16':DIP/'final.ckpt'}
    if not all(p.is_file() for p in checkpoints.values()):raise FileNotFoundError('Final checkpoints missing')
    manifest=dict(sequence='08',expected_full_frames=4071,generation_seeds=[42],
         cfg=[1,2],condition='100',gt_dir='gt_possion',sampler='Euler10',precision='fp16',physical_gpu=3,
         checkpoints={k:str(v) for k,v in checkpoints.items()},
         policy='ALL available frames, not B200 subset; exact seed42 for every frame; no reuse of previous seed/B200 metrics',
         start=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if args.resume:
        previous=read(root/'campaign_manifest.json')
        for key in ['sequence','expected_full_frames','generation_seeds','cfg','condition',
                    'gt_dir','sampler','precision','physical_gpu','checkpoints']:
            if previous.get(key)!=manifest[key]:
                raise ValueError(f'Resume protocol/checkpoint mismatch: {key}')
        if (root/'queue_status.json').exists():
            old_status=read(root/'queue_status.json')
            if old_status['status'] not in ['FAILED','COMPLETE']:
                raise ValueError('Resume only a stopped FAILED/COMPLETE campaign, not a possibly active queue')
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            save(root/f'queue_status_before_resume_{stamp}.json',old_status)
    else:
        save(root/'campaign_manifest.json',manifest)
    if args.inductor_beta0_fix:
        save(root/'numerical_runtime_recovery.json',dict(
            fix='pytorch20_inductor_baddbmm_beta0',precision='fp16',backend='inductor',
            scope='unfinished stages only; completed finite stages preserved',
            reason='GPU minimal reproduction: eager finite, original Inductor NaN, fixed Inductor exact to eager',
            changes_model_weights=False,changes_sampling_protocol=False))
    if args.cfg1_dense_backend=='eager':
        save(root/'cfg1_backend_exception.json',dict(
            cfg1_dense_backend='eager',cfg2_dense_backend='inductor',runtime_precision='fp16',
            scope='All three CFG1 methods, not only FPSGen; completed CFG2 preserved',
            reason='Repeated CFG1 compiled runtime NaN with finite inputs; same-input eager replays finite',
            diagnosis='Exact low-level cause unresolved; baddbmm probe fix did NOT resolve real evaluation',
            policy='Stop on any eager failure; no frame skipping or precision/seed changes'))
    completed=[]
    expected_frames=None
    # Complete the headline CFG2 for all models before the CFG1 appendix.
    for cfg in [2,1]:
        for model,ckpt in checkpoints.items():
            stage=f'{model}_cfg{cfg}';directory=root/model/f'cfg{cfg}'
            backend=args.cfg1_dense_backend if cfg==1 else 'inductor'
            if args.resume and (directory/'summary.json').exists():
                summary,frames,_=validate_full_result(directory,cfg,expected_frames,backend)
                if Path(summary['checkpoint']).resolve()!=ckpt.resolve():
                    raise ValueError('Completed result checkpoint path differs from campaign')
                expected_frames=frames
                completed.append(stage)
                print('REUSE VALIDATED',stage,flush=True)
                continue
            if args.resume:
                archived=archive_incomplete(root,model,cfg)
                if archived:print('PRESERVED FAILED ATTEMPT',archived,flush=True)
            gpu_free()
            command=[sys.executable,'-u','scripts/evaluation/eval_bevflow_checked.py','--bev-ckpt',str(ckpt),
                     '--dataset-root','/data-12/M2024-HWZ/KITTI_Odometry','--frame-stride','1',
                     '--output',str(directory),'--gt-dir','gt_possion','--generation-seeds','42','--base-seed','42',
                     '--guidance-scale',str(cfg),'--bev-steps','10','--runtime-precision','fp16',
                     '--dense-backend',backend,'--save-visuals','0','--progress-interval','100']
            if args.inductor_beta0_fix:
                command.append('--inductor-beta0-fix')
            save(root/'queue_status.json',dict(status='RUNNING',stage=stage,completed=completed,command=command))
            print('START',stage,flush=True)
            with (root/(stage+'.log')).open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
            if code:
                save(root/'queue_status.json',dict(status='FAILED',stage=stage,completed=completed,returncode=code))
                raise RuntimeError(f'{stage} failed; no automatic protocol changes')
            _,expected_frames,_=validate_full_result(directory,cfg,expected_frames,backend)
            completed.append(stage)
    summarize(root,checkpoints,args.cfg1_dense_backend)
    save(root/'queue_status.json',dict(status='COMPLETE',completed=completed,
         end=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('COMPLETE',root,flush=True)


if __name__=='__main__':main()
