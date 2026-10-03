#!/usr/bin/env python3
"""Separate final quality, condition usage, efficiency and paired uncertainty.

Generation seeds are averaged WITHIN each frame before resampling frames.
Missing/partial runs are never inserted as completed paper results.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS={"FPSGen BEVFlow":("legacy",None),"DiC-S-BEV":("dic_s_formal",2025),
         "SiD2-style Residual U-ViT-S-BEV":("sid2_s_formal",2025),"DiP-S/16-BEV":("dip_s16_formal",2026)}
METRICS=("occupancy_iou","occupancy_f1","completion_f1","height_mae_gtocc_m","density_mass_tv")


def paired_bootstrap(a,b,*,seed=20261001,resamples=10000,lower_is_better=False):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite matched frame arrays required")
    differences=a-b
    draws=np.random.default_rng(seed).integers(len(a),size=(resamples,len(a)))
    boot=differences[draws].mean(1)
    return {"mean_delta_ours_minus_baseline":float(differences.mean()),
            "ci95":np.quantile(boot,[.025,.975]).tolist(),
            "fraction_ours_gt_baseline":float((boot>0).mean()),
            "fraction_ours_better":float(((boot<0) if lower_is_better else (boot>0)).mean()),
            "fraction_policy":"fraction_ours_better is the paired-bootstrap distribution probability",
            "frame_fraction_ours_better":float(((differences<0) if lower_is_better else (differences>0)).mean()),
            "better_delta_sign":"negative" if lower_is_better else "positive",
            "frames":len(a),"resamples":resamples,"seed":seed}


def frame_means(path,expected_seeds=(20261001,20261002,20261003),expected_frames=None):
    grouped={}
    for row in csv.DictReader(path.open()):
        grouped.setdefault(row["frame"],[]).append(row)
    if len(grouped)!=200:
        raise ValueError(f"Final results require exactly200 frozen frames: {path}")
    if expected_frames is not None and set(grouped)!=set(expected_frames):
        raise ValueError("Final result frame identities differ from frozen B200 manifest")
    result={}
    for frame,rows in grouped.items():
        if len(rows)!=3 or sorted(int(r["seed"]) for r in rows)!=list(expected_seeds):
            raise ValueError("Final frame does not have the three exact generation seeds")
        values = np.asarray([[float(r[m]) for m in METRICS] for r in rows])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite final generation metrics")
        result[frame]={m:float(values[:,index].mean()) for index,m in enumerate(METRICS)}
    return result


def write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def validate_final_summary(summary, cfg):
    """Do not relabel a pilot, wrong sampler, or test-selected checkpoint final."""
    if summary.get('checkpoint_epoch') not in (499, 500):
        raise ValueError('Headline comparison requires completed epoch500 checkpoint')
    required = {'gt_dir': 'gt_possion', 'condition': '100', 'num_frames': 200,
                'samples_per_frame': 3, 'bev_steps': 10, 'guidance_scale': cfg,
                'generation_seeds': [20261001, 20261002, 20261003]}
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError('Final evaluation summary does not match frozen protocol')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("outputs/research_v2/formal_bev_compare"))
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    expected_frames=Path('configs/research_v2/formal_bev_compare/seq08_b200.txt').read_text().splitlines()
    quality=[];usage=[];efficiency=[];per_frame={};missing=[]
    plan=args.root/'campaign_plan.json'
    deferred_ids=json.loads(plan.read_text()).get('deferred_methods',[]) if plan.exists() else []
    deferred=[name for name,(directory,_) in METHODS.items() if directory.removesuffix('_formal') in deferred_ids]
    for method,(directory,year) in METHODS.items():
        root=args.root/directory
        if not all((root/f"cfg{cfg}/samples.csv").exists() for cfg in (1,2)):
            missing.append(method)
            continue
        row={"method":method,"year":year}
        per_frame[method]={}
        checkpoints=[]
        for cfg in (1,2):
            summary=json.loads((root/f"cfg{cfg}/summary.json").read_text())
            validate_final_summary(summary,cfg)
            checkpoints.append(summary['checkpoint_sha256'])
            rows=frame_means(root/f"cfg{cfg}/samples.csv",expected_frames=expected_frames)
            per_frame[method][cfg]=rows
            row.update({f"{m}_cfg{cfg}":float(np.mean([r[m] for r in rows.values()])) for m in METRICS})
        if len(set(checkpoints))!=1:
            raise ValueError('CFG1/CFG2 evaluation must use the same final checkpoint')
        quality.append(row)
        condition=json.loads((root/"condition_usage.json").read_text())
        usage.append({"method":method,"gshuffle_100_t02":condition["modes"]["100"]["0.2"]["g_shuffle"],
                      "gshuffle_100_t04":condition["modes"]["100"]["0.4"]["g_shuffle"],
                      **{f"gshuffle_{mode}_t02":condition["modes"][mode]["0.2"]["g_shuffle"] for mode in ("010","001","111")}})
        manifests=[json.loads(p.read_text()) for p in root.glob("run_manifest_rank*.json")]
        e={"method":method,"historical_provenance_incomplete":not bool(manifests)}
        reuse = root / 'reuse_manifest.json'
        if reuse.exists():
            evidence=json.loads(reuse.read_text())
            if evidence['epochs']!=500 or evidence['status']!='VERIFIED_COMPLETE_RETROSPECTIVE':
                raise ValueError('Historical reuse manifest is not verified epoch500')
            e.update(params_core=evidence['params']['core'],params_cond=evidence['params']['condition'],
                     params_pointpillar=evidence['params']['pointpillar'],params_total=evidence['params']['total'])
        if manifests:
            if len(manifests)!=2 or any(m.get('status')!='COMPLETE' or m.get('completed_epochs')!=500 for m in manifests):
                raise ValueError('New formal runs require completed500-epoch manifests from both ranks')
            e.update({"params_core":manifests[0]["params"]["core"],"params_cond":manifests[0]["params"]["condition"],
                      "params_pointpillar":manifests[0]["params"]["pointpillar"],"params_total":manifests[0]["params"]["total"],
                      "train_peak_allocated_mb":max(m["peak_allocated_mb"] for m in manifests),
                      "train_peak_reserved_mb":max(m["peak_reserved_mb"] for m in manifests)})
            e['train_backend']='inductor' if manifests[0]['compile_requested'] else 'eager'
            e['activation_checkpointing']=manifests[0].get('activation_checkpointing',False)
            e['grad_scaler_initial_scale']=manifests[0].get('grad_scaler_initial_scale')
        elif not reuse.exists():
            raise ValueError('Final quality requires rank manifests or a verified historical reuse manifest')
        if 'params_total' in e:
            row['params']=e['params_total']
        profile=root/"throughput.csv"
        if profile.exists():
            all_rows=list(csv.DictReader(profile.open()))
            if all_rows:
                e['train_first100_startup_inclusive_wall_sec']=float(all_rows[0]['interval_wall_ms'])/1000
            rows=[r for r in all_rows if 100<int(r["global_step"])<=400]
            if len(rows)==3:
                wall=sum(float(r["interval_wall_ms"]) for r in rows)/1000
                e.update(train_global_samples_sec=9600/wall,train_ms_per_optimizer_step=wall*1000/300)
        infer=args.root/({"dic_s_formal":"dic_s","sid2_s_formal":"sid2_s","dip_s16_formal":"dip_s16"}.get(directory,directory))/"inference_efficiency.json"
        if infer.exists():
            result=json.loads(infer.read_text())
            e.update(inference_backend=result['backend'],inference_b1_ms=result["batches"]["1"]["mean_ms_per_sample"],
                     first_generation_including_compile_sec=result['batches']['1']['first_generation_including_compile_sec'],
                     inference_b8_samples_sec=result["batches"]["8"]["samples_per_second"])
        efficiency.append(e)
    comparisons={}
    if "FPSGen BEVFlow" in per_frame:
        for method in per_frame:
            if method=="FPSGen BEVFlow":continue
            comparisons["FPSGen - "+method]={}
            for cfg in (1,2):
                a,b=per_frame["FPSGen BEVFlow"][cfg],per_frame[method][cfg]
                if set(a)!=set(b):raise ValueError("Paired comparison has mismatched frame identities")
                frames=sorted(a)
                comparisons["FPSGen - "+method][f"cfg{cfg}"]={m:paired_bootstrap([a[f][m] for f in frames],
                    [b[f][m] for f in frames],lower_is_better=m in METRICS[-2:]) for m in METRICS}
    for name,rows in (("final_quality.csv",quality),("final_condition_usage.csv",usage),("final_efficiency.csv",efficiency)):
        write_csv(args.root/name,rows)
    (args.root/"paired_bootstrap.json").write_text(json.dumps({"policy":"Average3 seeds within each frame, then paired frame bootstrap",
        "complete":not missing,"missing_models":missing,"deferred_models":deferred,
        "selected_scope_complete":not [method for method in missing if method not in deferred],
        "comparisons":comparisons},indent=2)+"\n")
    print(json.dumps({"complete":not missing,"missing_models":missing,"deferred_models":deferred,
        "selected_scope_complete":not [method for method in missing if method not in deferred]},indent=2))


if __name__=="__main__":
    main()
