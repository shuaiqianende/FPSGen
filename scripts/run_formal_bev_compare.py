#!/usr/bin/env python3
"""GPU2+3 serial comparison. Every gate is mandatory; first failure stops.

Fresh preflights/pilots/formal runs use independent output directories and
random initialization. Nothing resumes, overwrites a checkpoint, kills an
existing GPU process, or silently changes batch/precision/architecture.
"""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/"configs/research_v2/formal_bev_compare"
OUTPUT = ROOT/"outputs/research_v2/formal_bev_compare"
METHODS = ("dic_s", "sid2_s", "dip_s16")
EXPECTED_SOURCE_HASH = None
LEGACY = ROOT/"experiments/fpsgen_bevflow_legacy_gt_possion_formal_500ep_b16_gpu23_fp16_inductor/lightning_logs/version_0/checkpoints/fpsgen_bevflow_legacy_gt_possion_formal_500ep_b16_gpu23_fp16_inductor_epoch=499.ckpt"


def selected_config_path(method, pilot=False):
    # Explicitly selected AFTER logged failure and successful eager/DDP
    # revalidation; this is not an automatic try/except backend fallback.
    if method == 'dic_s':
        suffix='10ep_pilot_eager_checkpointed' if pilot else '500ep_ddp2_b16_fp16_eager_checkpointed'
    else:
        suffix='10ep_pilot' if pilot else '500ep_ddp2_b16_fp16_inductor'
    return CONFIGS/f'{method}_{suffix}.yaml'


def assert_free():
    # Inspect exact physical GPUs, not inferred utilization alone: an idle
    # process can still hold a context. Never stop another process.
    ids = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuids = {line.split(",")[1].strip() for line in ids.splitlines() if int(line.split(",")[0]) in (2,3)}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = [line for line in apps.splitlines() if line.split(",")[0].strip() in uuids]
    if busy:
        raise RuntimeError(f"GPU2/3 occupied, STOP (no kill): {busy}")


def environment(visible="2,3"):
    result = os.environ.copy()
    result["CUDA_VISIBLE_DEVICES"] = visible
    result["PYTHONPATH"] = str(ROOT)+os.pathsep+result.get("PYTHONPATH", "")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        result[name] = "1"
    for name, directory in {"TMPDIR":"env/tmp", "MPLCONFIGDIR":"env/mpl_cache",
                            "KEOPS_CACHE_FOLDER":"env/keops_cache_formal_compare",
                            "TORCHINDUCTOR_CACHE_DIR":"env/inductor_cache_formal_compare"}.items():
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
        result[name] = str(ROOT/directory)
    result["TRAIN_DATABASE"] = os.environ.get("TRAIN_DATABASE", "/data-12/M2024-HWZ/KITTI_Odometry")
    return result


def freeze_frames(data_root, count):
    candidates = sorted({p.stem for p in (Path(data_root)/"08/gt_possion").iterdir()
                         if p.suffix in (".npy", ".ply") and (Path(data_root)/"08/input_"/(p.stem+".npy")).is_file()})
    if len(candidates) < count:
        raise ValueError("Insufficient GT/input pairs for frozen manifest")
    frames = [candidates[i] for i in np.rint(np.linspace(0,len(candidates)-1,count)).astype(int)]
    if len(set(frames)) != count:
        raise ValueError("Uniform manifest must contain unique frames")
    path = CONFIGS/f"seq08_b{count}.txt"
    content = "\n".join(frames)+"\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError("Existing frozen frame manifest differs; refusing replacement")
    if not path.exists():
        path.write_text(content)
    return path


def run_command(command, label, visible="2,3"):
    assert_free()
    if EXPECTED_SOURCE_HASH is not None:
        from fpsgen.utils.formal_bev import source_fingerprint
        if source_fingerprint(ROOT)['sha256']!=EXPECTED_SOURCE_HASH:
            raise RuntimeError('Local Python sources changed after campaign freeze; STOP before next stage, do not mix versions')
    directory = OUTPUT/label
    if directory.exists():
        raise FileExistsError(f"Refusing repeated stage/overwritten evidence: {directory}")
    directory.mkdir(parents=True)
    print(f"\n===== START {label} GPU={visible} =====",flush=True)
    state={"status":"RUNNING", "stage":label, "command":command,
           "start":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (OUTPUT/"queue_status.json").write_text(json.dumps(state,indent=2)+"\n")
    formal_sessions={'dic_s_formal':'fpsgen_formal_dic', 'sid2_s_formal':'fpsgen_formal_sid2',
                     'dip_s16_formal':'fpsgen_formal_dip'}
    session=formal_sessions.get(label)
    if session:
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
            raise RuntimeError(f'Existing tmux session {session}; STOP without replacing it')
        runtime_env=environment(visible)
        safe_keys=('CUDA_VISIBLE_DEVICES','PYTHONPATH','TRAIN_DATABASE','OMP_NUM_THREADS',
                   'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','TMPDIR',
                   'MPLCONFIGDIR','KEOPS_CACHE_FOLDER','TORCHINDUCTOR_CACHE_DIR')
        spec=directory/'worker_spec.json'
        spec.write_text(json.dumps({'command':command,'directory':str(directory),'cwd':str(ROOT),
            'environment':{key:runtime_env[key] for key in safe_keys}},indent=2)+'\n')
        worker=shlex.join([sys.executable,'-u',str(ROOT/'scripts/run_formal_bev_stage.py'),'--spec',str(spec)])
        subprocess.run(['tmux','new-session','-d','-s',session,worker+'; exec bash'],check=True)
        print(f'Attach active formal run: tmux attach -t {session}',flush=True)
        state['tmux_session']=session
        (OUTPUT/'queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
        stream=None
        try:
            while True:
                if stream is None and (directory/'console.log').exists():
                    stream=(directory/'console.log').open()
                if stream is not None:
                    new=stream.read()
                    if new: print(new,end='',flush=True)
                result=directory/'worker_result.json'
                if result.exists():
                    if stream is not None:
                        print(stream.read(),end='',flush=True)
                    code=json.loads(result.read_text())['returncode']
                    break
                if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:
                    raise RuntimeError('Formal tmux disappeared without a result; STOP, no orphan-process kill')
                time.sleep(1)
        finally:
            if stream is not None:stream.close()
    else:
        with (directory/"console.log").open("w") as log:
            process=subprocess.Popen(command,cwd=ROOT,env=environment(visible),stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                log.write(line)
                print(line,end="",flush=True)
            code=process.wait()
    state.update(status="PASS" if code==0 else "FAILED",returncode=code,
                 end=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (directory/"stage_status.json").write_text(json.dumps(state,indent=2)+"\n")
    (OUTPUT/"queue_status.json").write_text(json.dumps(state,indent=2)+"\n")
    if code:
        raise RuntimeError(f"{label} failed; queue STOP, no fallback/next experiment")


def stage_config(base, label, *, gpus=2, compile_dense=None, max_steps=None):
    if (OUTPUT/label).exists():
        raise FileExistsError(f'Refusing overwritten stage/config evidence: {OUTPUT/label}')
    cfg=copy.deepcopy(base)
    cfg["experiment"]["id"]="fpsgen_formal_compare_"+label
    cfg["train"]["n_gpus"]=gpus
    if max_steps:
        cfg["train"]["max_steps"]=max_steps
    cfg["runtime"].update(compile_dense=bool(base['runtime'].get('compile_dense')) if compile_dense is None else compile_dense,
        formal_manifest_dir=str(OUTPUT/label), throughput_csv=str(OUTPUT/label/"throughput.csv"))
    path=OUTPUT/"generated_configs"/(label+".yaml")
    if path.exists():
        raise FileExistsError(f'Refusing overwritten generated config: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return path


def train(base, label, *, gpus=2, compile_dense=None, preflight=0, max_steps=None):
    path=stage_config(base,label,gpus=gpus,compile_dense=compile_dense,max_steps=max_steps)
    command=[sys.executable,"fpsgen/train_bev.py","-c",str(path)]
    if preflight:
        command += ["--preflight-steps",str(preflight)]
    run_command(command,label,"2,3" if gpus==2 else "2")
    return OUTPUT/label/"final.ckpt"


def evaluate(ckpt, label, pilot, data_root):
    import torch
    metadata=torch.load(ckpt,map_location="cpu")
    backend="inductor" if metadata["hyper_parameters"].get("runtime",{}).get("compile_dense",False) else "eager"
    del metadata
    condition_manifest=freeze_frames(data_root,100 if pilot else 200)
    run_command([sys.executable,"scripts/eval_formal_bev_condition_usage.py","--checkpoint",str(ckpt),
                 "--data-root",data_root,"--manifest",str(condition_manifest),
                 "--output",str(OUTPUT/label/"condition_usage.json")],label+"_condition_eval","2")
    usage=json.loads((OUTPUT/label/"condition_usage.json").read_text())
    if pilot:
        for mode in ("100","111"):
            gain=usage["modes"][mode]["0.2"]["g_shuffle"]
            if gain < .005:
                raise RuntimeError(f"Pilot condition-use needs inspection: mode={mode}, Gshuffle={gain}; STOP")
    manifest=freeze_frames(data_root,20 if pilot else 200)
    for cfg in (1,2):
        run_command([sys.executable,"scripts/eval_bevflow_lidar_only.py", "--bev-ckpt",str(ckpt),
            "--dataset-root",data_root,"--manifest",str(manifest),"--output",str(OUTPUT/label/f"cfg{cfg}"),
            "--generation-seeds",*(str(s) for s in ((20261001,) if pilot else (20261001,20261002,20261003))),
            "--guidance-scale",str(cfg),"--save-visuals","0","--progress-interval","10",
            "--bev-steps","10","--runtime-precision","fp16","--dense-backend",backend],label+f"_cfg{cfg}_eval","2")


def verify_legacy():
    import torch
    state=torch.load(LEGACY,map_location="cpu")
    cfg=state["hyper_parameters"]
    wanted=yaml.safe_load((CONFIGS/"fpsgen_legacy_500ep_ddp2_b16_fp16_inductor.yaml").read_text())
    for section,keys in {"train":("n_gpus","batch_size","max_epoch","seed","lr","num_workers","prefetch_factor","pin_memory","persistent_workers"),
                         "data":("gt_dir","num_points","max_range","train"),
                         "runtime":("precision","compile_dense","compile_backend")}.items():
        for key in keys:
            if cfg[section][key]!=wanted[section][key]:
                raise ValueError(f"Legacy reuse protocol mismatch {section}.{key}")
    if state["epoch"]!=499 or state["global_step"]!=299000:
        raise ValueError("Legacy not the completed 500-epoch final")
    for group in state['optimizer_states'][0]['param_groups']:
        if tuple(group['betas'])!=(.9,.999) or group['weight_decay']!=.0001:
            raise ValueError('Historical Legacy optimizer differs from common AdamW contract')
    from fpsgen.models.bev_backbones.factory import build_bev_backbone
    from fpsgen.utils.formal_bev import parameter_counts
    params=parameter_counts(build_bev_backbone(cfg))
    directory=OUTPUT/"legacy"
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"reuse_manifest.json").write_text(json.dumps({"status":"VERIFIED_COMPLETE_RETROSPECTIVE",
        "checkpoint":str(LEGACY),"epochs":500,"steps":state["global_step"],"config":dict(cfg),
        "params":params,
        "historical_skipped_optimizer_steps":None,"retrospective_limitations":"Historical scaler skip count and initial dirty source hash unavailable; do not claim zero overflows or complete original runtime provenance.",
        "checkpoint_cadence_difference":"Historical run saved every epoch then pruned to every10; new runs save every50. This affects checkpoint I/O but not optimizer/task contract."},indent=2)+"\n")


def main():
    global OUTPUT, EXPECTED_SOURCE_HASH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUTPUT,
                        help='Fresh campaign directory under repository outputs; old failures are never overwritten')
    parser.add_argument("--preflight-only",action="store_true",help="Only eager/single/DDP model gates, no pilots/formal training")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--skip-legacy-eval",action="store_true",help="Defer Legacy final evaluation explicitly; never implies completion")
    parser.add_argument("--diagnostic-stage",choices=("single-inductor","single-eager","ddp","pipeline"),
                        help="Explicit isolated diagnostic after a stopped queue; never starts pilot/formal")
    parser.add_argument("--activation-checkpointing",action="store_true",
                        help="Explicit isolated DiC B16 diagnostic after confirmed OOM; not a silent formal change")
    parser.add_argument("--diagnostic-tag",default="",help="New evidence suffix for an explicit diagnostic retry")
    parser.add_argument("--diagnostic-backend",choices=("eager","inductor"),default="inductor",
                        help="Explicit DDP diagnostic backend exception; never silent fallback")
    parser.add_argument("--diagnostic-init-scale",type=float,
                        help="Explicit FP16 scaler calibration diagnostic; does not change formal configs")
    parser.add_argument("--diagnostic-steps",type=int,
                        help="Additional isolated cold-start investigation, not a replacement for the required16/32-step gates")
    args=parser.parse_args()
    OUTPUT=args.output_root.resolve()
    if not OUTPUT.is_relative_to(ROOT/'outputs'):
        raise ValueError('Campaign evidence must remain inside repository outputs')
    if not args.diagnostic_stage and not args.preflight_only and set(args.methods)!=set(METHODS):
        raise ValueError('Formal main-table queue cannot silently omit a failed candidate; use diagnostics to investigate')
    if not args.diagnostic_stage and (args.activation_checkpointing or args.diagnostic_init_scale is not None or args.diagnostic_steps):
        raise ValueError('Diagnostic overrides must not silently modify a formal run')
    if args.diagnostic_init_scale is not None and args.diagnostic_init_scale<=0:
        raise ValueError('Positive GradScaler initial scale required')
    if args.diagnostic_steps is not None and args.diagnostic_steps<=0:
        raise ValueError('Positive diagnostic step count required')
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    assert_free()
    verify_legacy()
    data_root=environment()["TRAIN_DATABASE"]
    freeze_frames(data_root,200)
    if args.diagnostic_stage:
        if args.diagnostic_stage == "pipeline":
            cfg=yaml.safe_load((CONFIGS/"fpsgen_legacy_500ep_ddp2_b16_fp16_inductor.yaml").read_text())
            cfg["runtime"]["formal_stall_gate"]=True
            train(cfg,"legacy_ddp_pipeline_gate_v2",max_steps=400)
        else:
            for method in args.methods:
                cfg=yaml.safe_load(selected_config_path(method).read_text())
                if args.diagnostic_init_scale is not None:
                    cfg["runtime"]["grad_scaler_init_scale"]=args.diagnostic_init_scale
                if args.activation_checkpointing:
                    if method != "dic_s":
                        raise ValueError("Only DiC's checkpointing seam has been validated")
                    cfg["runtime"]["activation_checkpointing"]=True
                ddp=args.diagnostic_stage=="ddp"
                compiled=(args.diagnostic_backend=="inductor" if ddp else args.diagnostic_stage=="single-inductor")
                train(cfg,method+("_ddp_b16" if ddp else ("_inductor_b16" if compiled else "_eager_b16"))+("_checkpointed" if args.activation_checkpointing else "")+args.diagnostic_tag,gpus=2 if ddp else 1,compile_dense=compiled,
                      preflight=args.diagnostic_steps or (32 if ddp else 16))
        return
    from fpsgen.utils.formal_bev import source_fingerprint
    frozen_formal={method:yaml.safe_load(selected_config_path(method).read_text()) for method in args.methods}
    frozen_pilot={method:yaml.safe_load(selected_config_path(method,pilot=True).read_text()) for method in args.methods}
    fingerprint=source_fingerprint(ROOT)
    EXPECTED_SOURCE_HASH=fingerprint['sha256']
    plan=OUTPUT/'campaign_plan.json'
    if plan.exists():
        raise FileExistsError('Campaign plan already exists; select a fresh output-root instead of resuming/overwriting')
    plan.write_text(json.dumps({'start_time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'local_source_fingerprint':fingerprint,'selected_formal_configs':frozen_formal,
        'selected_pilot_configs':frozen_pilot,'gpu_visibility':'2,3',
        'policy':'Fresh all pilots before any new formal500; first failure stops; source/config plan frozen'},indent=2)+'\n')
    # Exact production loader/runtime on both GPUs. First100 steps include
    # compile startup, next300 constitute the stable host-gap/throughput gate.
    legacy=yaml.safe_load((CONFIGS/"fpsgen_legacy_500ep_ddp2_b16_fp16_inductor.yaml").read_text())
    legacy["runtime"]["formal_stall_gate"]=True
    legacy['runtime'].update(preflight_validation=True,preflight_only=False,
                             preflight_steps=32,expected_batch_size=16)
    train(legacy,"legacy_ddp_pipeline_gate",max_steps=400)
    for method in args.methods:
        formal=frozen_formal[method]
        train(formal,method+"_eager_b16",gpus=1,compile_dense=False,preflight=16)
        if formal['runtime'].get('compile_dense'):
            train(formal,method+"_inductor_b16",gpus=1,preflight=16)
        else:
            reason=formal['runtime'].get('backend_exception')
            proof=ROOT/formal['runtime'].get('compile_failure_stage','')
            if not reason or not proof.is_file() or json.loads(proof.read_text())['status']!='FAILED':
                raise RuntimeError('Eager formal backend requires an explicit documented failed compile stage')
            print(f'EXPLICIT BACKEND EXCEPTION {method}: {reason}',flush=True)
        train(formal,method+"_ddp_b16",preflight=32)
    if args.preflight_only:
        print("ALL MODEL PREFLIGHTS PASS; pilots/formal NOT STARTED",flush=True)
        return
    # Validate all candidates before investing in any 500-epoch run.
    for method in args.methods:
        pilot=frozen_pilot[method]
        ckpt=train(pilot,method+"_pilot")
        evaluate(ckpt,method+"_pilot",True,data_root)
    for method in args.methods:
        formal=frozen_formal[method]
        train(formal,method+"_formal_preflight",preflight=32)
        # Fresh seed42 initialization, never --checkpoint/--weights resume.
        ckpt=train(formal,method+"_formal")
        evaluate(ckpt,method+"_formal",False,data_root)
        run_command([sys.executable,"scripts/benchmark_formal_bev_inference.py","--checkpoint",str(ckpt),
            "--data-root",data_root,"--manifest",str(CONFIGS/"seq08_b200.txt"),
            "--output",str(OUTPUT/method/"inference_efficiency.json"),
            "--dense-backend",'inductor' if formal['runtime'].get('compile_dense') else 'eager'],method+"_inference_benchmark","2")
    if not args.skip_legacy_eval:
        evaluate(LEGACY,"legacy",False,data_root)
        run_command([sys.executable,"scripts/benchmark_formal_bev_inference.py","--checkpoint",str(LEGACY),
            "--data-root",data_root,"--manifest",str(CONFIGS/"seq08_b200.txt"),
            "--output",str(OUTPUT/"legacy/inference_efficiency.json")],"legacy_inference_benchmark","2")
    subprocess.run([sys.executable,"scripts/summarize_formal_bev_compare.py","--root",str(OUTPUT)],cwd=ROOT,check=True)
    print("QUEUE COMPLETE (requested stages only)",flush=True)


if __name__=="__main__":
    main()
