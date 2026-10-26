"""Fresh scratch gate -> discarded32-update smoke -> fresh formal20K, one GPU per arm."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,gc,json,os,subprocess,time,traceback
import torch
from fpsgen.hier_occ_flow.queued20k_training import (load_config,own_sha as own_source_sha,admission,QueueSystem as ScratchSystem,QueueState as RunState,run)
from fpsgen.hier_occ_flow.scratch20k_training import cpu_state,exact_state
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import datasets,config_sha
from fpsgen.hier_occ_flow.meanflow_replay_bounds import compare_native_replay,POLICY_SHA
from fpsgen.hier_occ_flow.storage import atomic_json
from fpsgen.hier_occ_flow.trainer import seed_all


def engineering(cfg,g,out):
    admission_report=admission(cfg)
    if out.exists():raise FileExistsError('Preserve prior engineering gate')
    out.mkdir(parents=True)
    report=dict(status='RUNNING',source_sha256=own_source_sha(),config_sha256=config_sha(cfg),gpu=admission_report,
        replay_policy_sha256=POLICY_SHA,initialization_mode='scratch',physical_gpu=cfg['physical_gpu'],true_batch=2,truncation=False)
    start=time.time()
    try:
        train,dev,manifests=datasets(cfg,g,verified_receipt=cfg['verified_data_receipt'])
        selected=sorted([(int(r['length']),split,i) for split,ds in [('train',train),('development',dev)] for i,r in enumerate(ds.index)],reverse=True)[:2]
        records=[(train if split=='train' else dev)[i] for n,split,i in selected]
        # Dense Stage1 always uses the full384x384x32 grid, and scans are never subsampled.
        # Stage2 selects the global largest GT support pair and includes all25%negatives.
        state=RunState(cfg,g,'cuda');report['initialization']=state.model.initialization
        torch.cuda.reset_peak_memory_stats();initial=state.model.initialization['initial_state_sha256']
        first,_=state.update(records,force_finite=True)
        if not first['finite'] or any(v<=0 or not __import__('math').isfinite(v) for v in first['gradient_modules'].values()):raise ValueError('Every trained component needs finite nonzero gradient')
        if not state.model.is_s1 and first['scene_lengths']!=[n+n//4 for n,split,i in selected]:raise ValueError('Largest full support bound not attained')
        if first['jvp_calls']!=0:raise ValueError('No MeanFlow JVP in FM/Shortcut followups')
        if state.model.is_shortcut and (first['teacher_flow_forwards']!=2 or first['interval_last_gradient_norm']<=0):raise ValueError('Two-half-step EMA target/step embedding gradient not exercised')
        if initial==__import__('fpsgen.hier_occ_flow.encoder_adaptation',fromlist=['tensor_state_sha']).tensor_state_sha(state.model.state_dict()):raise ValueError('No parameter change')
        saved=state.snapshot();torch.save(saved,out/'checkpoint_audit.pt');loaded=torch.load(out/'checkpoint_audit.pt',map_location='cpu');state.restore(loaded)
        if not exact_state(saved,state.snapshot()):raise ValueError('Disk checkpoint/RNG/EMA/Adam/Scaler roundtrip not exact')
        expected_row,expected=state.update(records,force_finite=True,capture=True);left=dict(**state.snapshot(),**expected)
        state.restore(loaded)
        if not exact_state(loaded,state.snapshot()):raise ValueError('Before replay state mismatch')
        actual_row,actual=state.update(records,force_finite=True,capture=True);right=dict(**state.snapshot(),**actual)
        bounds=compare_native_replay(left,right)
        report['native_resume']=dict(checkpoint_roundtrip_exact=True,bounds=bounds)
        if not expected_row['finite'] or not actual_row['finite'] or bounds['status']!='PASS':raise ValueError('Native scratch update replay fails fixed bounds')
        if not state.model.is_s1:
            actual_frozen=__import__('fpsgen.hier_occ_flow.encoder_adaptation',fromlist=['tensor_state_sha']).tensor_state_sha(state.model.frozen_condition.model.state_dict())
            if actual_frozen!=state.model.frozen_condition.audit['initial_state_sha256']:raise ValueError('Frozen Encoder parameters or buffers changed')
            report['frozen_encoder_unchanged']=True
        report.update(status='GPU_PASS',manifests=manifests,first_update=first,stress_selection=selected,
            scenes=[dict(sequence=r['sequence'],frame=r['frame'],positive_parents=len(r['parent_ids'])) for r in records],
            native_jvp_exercised=False,shortcut_teacher_forwards=first['teacher_flow_forwards'],resume_classification='EXACT_NATIVE_REPLAY' if bounds['exact'] else 'NUMERICAL_NATIVE_REPLAY_WITHIN_FIXED_BOUNDS')
    except Exception as exc:
        report.update(status='BLOCKED_ENGINEERING',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(out/'result.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','first_update','reason','peak_allocated_bytes')},indent=2),flush=True)
    if report['status']!='GPU_PASS':raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','smoke','formal','pipeline'),required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    cfg,g=load_config(a.config);root=Path(cfg['output_root'])
    if a.mode=='gate':engineering(cfg,g,root/'engineering');return
    if a.mode in ('smoke','formal'):run(cfg,g,root/a.mode,smoke=a.mode=='smoke',resume=a.resume);return
    if root.exists():raise FileExistsError('Fresh pipeline output required')
    root.mkdir(parents=True);pin=own_source_sha();cpin=config_sha(cfg)
    atomic_json(root/'pipeline_launch.json',dict(pid=os.getpid(),epoch=time.time(),source_sha256=pin,config_sha256=cpin,initialization_mode='scratch',pretrained_sources=False))
    try:
        for mode in ('gate','smoke','formal'):
            if own_source_sha()!=pin:raise ValueError('Pinned source changed')
            atomic_json(root/'supervisor.json',dict(status='RUNNING_'+mode.upper(),pid=os.getpid(),physical_gpu=cfg['physical_gpu'],arm=cfg['arm'],source_sha256=pin,initialization_mode='scratch'))
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(a.config.resolve()),'--mode',mode]
            with (root/(mode+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(root/'supervisor.json',dict(status='RUNNING_'+mode.upper(),pid=os.getpid(),child_pid=child.pid,physical_gpu=cfg['physical_gpu'],arm=cfg['arm'],source_sha256=pin,command=command,initialization_mode='scratch'))
                code=child.wait()
            if code:raise RuntimeError(f'{mode} exited{code}; no automatic fallback/retry')
        summary=json.loads((root/'formal/summary.json').read_text())
        if summary['status']!='TRAINING_COMPLETE' or summary['counts']['exposures']!=20000:raise ValueError('Formal budget incomplete')
        atomic_json(root/'supervisor.json',dict(status='RUNNING_QUALITY41',pid=os.getpid(),arm=cfg['arm'],automatic_extra_training=False))
        command=[sys.executable,'-u',str(ROOT/'scripts/research_v2/queued20k/evaluate.py'),'--config',str(a.config.resolve()),'--checkpoint',str(root/'formal/checkpoints/resume_latest.pt'),'--output',str(root/'evaluation/seq08_41')]
        with (root/'quality41.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(root/'supervisor.json',dict(status='RUNNING_QUALITY41',pid=os.getpid(),child_pid=child.pid,arm=cfg['arm'],command=command))
            code=child.wait()
        if code:raise RuntimeError(f'quality41 exited{code}; preserve partial results')
        atomic_json(root/'supervisor.json',dict(status='COMPLETE20K_AND_SEQ08_41',pid=os.getpid(),arm=cfg['arm'],automatic_extra_training=False))
    except Exception as exc:
        atomic_json(root/'supervisor.json',dict(status='BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),pid=os.getpid(),arm=cfg['arm']));raise


if __name__=='__main__':main()
