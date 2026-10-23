"""Run the authorized single-seed B0/M1 20K pair, then evaluate; no extra training arms."""
from common import ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha

PYTHON='/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python'
SCRIPTS=ROOT/'scripts/research_v2/hier_occ_flow'
CONFIGS=ROOT/'configs/research_v2/hier_occ_flow'


def health_report(folder, expected):
    """Successful counters, real B2, finite gradients and recent AMP stability."""
    summary=json.loads((folder/'summary.json').read_text())
    rows=[json.loads(x) for x in (folder/'train_rank0.jsonl').read_text().splitlines() if x.strip()]
    window=[r for r in rows if r['successful_updates']<=expected]
    successful={r['successful_updates']:r for r in window if r['finite']}
    recent=[successful[k] for k in sorted(successful)[-min(100,expected):]]
    checks={
        'successful_updates':summary['successful_updates']>=expected and expected in successful,
        'successful_exposures':summary['successful_frame_exposures']==summary['successful_updates']*2,
        'batch2':bool(recent) and all(r['global_batch']==2 and r['batch_per_gpu']==2 for r in recent),
        'finite_positive_gradients':bool(recent) and all(r['loss'] is not None and r['gradient_norm'] is not None
            and r['gradient_norm']>0 for r in recent),
        'recent_attempts_finite':all(r['finite'] for r in window[-20:]),
        'amp_skip_fraction_below_5pct':window[-1]['amp_skips']/max(1,window[-1]['attempts'])<=.05,
    }
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'checked_update':expected,'successful_updates':summary['successful_updates'],'successful_frame_exposures':summary['successful_frame_exposures'],
        'recent_mean_loss':sum(r['loss'] for r in recent)/max(1,len(recent)),
        'recent_mean_update_seconds':sum(r['seconds'] for r in recent)/max(1,len(recent)),
        'amp_skips':rows[-1]['amp_skips'],'scope':'engineering/learning health, not a quality selection',
        'epoch':time.time()}
    atomic_json(folder/f'health_{expected:06d}.json',result)
    if result['status']!='PASS':raise RuntimeError(f'Health check failed: {folder}: {checks}')
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--verification-pid',type=int)
    p.add_argument('--review-decision',type=Path,required=True)
    p.add_argument('--source-worktree',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen_source=source_sha()
    gpu_map={'b0':1,'m1':3}  # User explicitly authorized GPU1 and GPU3, 2026-10-24.
    atomic_json(a.output/'queue_contract.json',{
        'supervisor_pid':os.getpid(),'source_sha256':frozen_source,'physical_gpus':gpu_map,
        'training_seed':20261032,'batch_per_gpu':2,'successful_frame_exposures_per_arm':20000,
        'successful_updates_per_arm':10000,'initialization':'same raw-GT source, additional matched adaptation',
        'staged_updates':[100,500,10000],'formal_arms':['b0','m1'],
        'extra_training_seeds':False,'automatic_100k':False,'epoch':time.time()})
    gate_path=ROOT/'experiments/desescene_state_condition_v1/gate.json'
    if not gate_path.exists():
        inherited=a.source_worktree/'experiments/desescene_v_next/engineering_gate_child_bridge.json'
        gate=json.loads(inherited.read_text())
        gate.update(research_model_smoke=False,state_condition_gpu_verified=False,c1_review_complete=False,
                    inherited_engineering_gate_sha256=file_sha(inherited))
        atomic_json(gate_path,gate)

    def run(name,gpu,script,args,completion,expected_updates=None):
        if source_sha()!=frozen_source:raise RuntimeError('Source changed: stop before launching another task')
        status_path=a.output/(name+'.status.json')
        if completion.exists():
            value=json.loads(completion.read_text())
            if value.get('status') in ('FAIL','FAILED','BLOCKED'):raise RuntimeError('Retained failed task '+name)
            if expected_updates is None or value.get('successful_updates',0)>=expected_updates:return
        while gpu_preflight((gpu,))['status']!='AVAILABLE':
            atomic_json(status_path,{'status':'WAITING_GPU_IDLE','physical_gpu':gpu,'epoch':time.time()})
            time.sleep(20)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='2',HS_STAGE2_ALLOW_SHARED_COMPUTE='0',
                 TMPDIR='/data-12/M2024-HWZ/FPSGen/env/tmp',
                 TORCH_EXTENSIONS_DIR='/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/torch_extensions')
        cmd=[('bash' if script.suffix=='.sh' else PYTHON),str(script),*map(str,args)]
        with (a.output/(name+'.log')).open('a',buffering=1) as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(status_path,{'status':'RUNNING','pid':process.pid,'command':cmd,'physical_gpu':gpu,'epoch':time.time()})
            code=process.wait()
        valid=code==0 and completion.exists()
        atomic_json(status_path,{'status':'COMPLETE' if valid else 'FAILED','command':cmd,'exit_code':code,'epoch':time.time()})
        if not valid:raise RuntimeError(name+' failed; inspect preserved log')
        value=json.loads(completion.read_text())
        if value.get('status') in ('FAIL','FAILED','BLOCKED'):raise RuntimeError(name+' did not pass')

    def smoke(arm):
        folder=a.output/('engineering_'+arm)
        run('engineering_'+arm,gpu_map[arm],SCRIPTS/'launch_stage2_384.sh',[
            '--config',CONFIGS/f'stage2_384_desescene_sci_{arm}_smoke.yaml',
            '--output',folder,'--max-updates',32],folder/'summary.json',32)
        return folder

    def train_and_evaluate(arm):
        gpu=gpu_map[arm];folder=a.output/(arm+'_20k')
        for stop in (100,500,10000):
            args=['--config',CONFIGS/f'stage2_384_desescene_sci_{arm}_20k.yaml','--output',folder,'--max-updates',stop]
            checkpoint=folder/'checkpoints/resume_latest.pt'
            if checkpoint.exists():args+=['--resume',checkpoint]
            run(f'{arm}_train_{stop}',gpu,SCRIPTS/'launch_stage2_384.sh',args,folder/'summary.json',stop)
            health_report(folder,stop)
        primary=folder/'evaluation/primary900_nfe4.json'
        run(arm+'_evaluate900',gpu,SCRIPTS/'eval_stage2_384.py',[
            '--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt',
            '--output',primary,'--ema','--nfes',4,'--max-frames',900,'--sampling-seeds',20261031,
            '--continuous-metrics'],primary)
        mechanism=folder/'evaluation/mechanisms64'
        run(arm+'_mechanisms64',gpu,SCRIPTS/'audit_desescene_mechanisms.py',[
            '--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt',
            '--output',mechanism,'--max-frames',64,'--mode','both','--ema'],mechanism/'summary.json')
        if arm=='m1':
            order=('r0_gpu2_20k','e1_gpu3_20k','e2_gpu2_20k','e3_gpu3_20k')
            for round_id in range(1,4):
                for name in (tuple(reversed(order)) if round_id==2 else order):
                    trained=a.source_worktree/'experiments/dense_fidelity_v2_single_gpu'/name
                    output=ROOT/'experiments/desescene_state_condition_v1/phase0/d4_encoder'/f'{name}_round{round_id}.json'
                    run(f'd4_{name}_{round_id}',gpu,SCRIPTS/'benchmark_trained_encoder_pilot.py',[
                        '--config',trained/'config.json','--checkpoint',trained/'checkpoints/step010000.pt',
                        '--output',output,'--physical-gpu',gpu,'--rounds',1,'--precision','explicit_fp16'],output)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            smokes=list(pool.map(smoke,('b0','m1')))
        while not a.verification.exists():
            if a.verification_pid:
                try:os.kill(a.verification_pid,0)
                except ProcessLookupError:raise RuntimeError('GPU verification stopped without a result')
            time.sleep(5)
        subprocess.run([PYTHON,str(SCRIPTS/'register_desescene_m1_gate.py'),'--gate',str(gate_path),
            '--gpu-verification',str(a.verification),'--smoke-outputs',*map(str,smokes),
            '--review-decision',str(a.review_decision)],cwd=ROOT,check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train_and_evaluate,arm) for arm in ('b0','m1')]
            failures=[]
            for future in futures:
                try:future.result()
                except Exception as exc:failures.append(str(exc))
            if failures:raise RuntimeError('; '.join(failures))
        atomic_json(a.output/'summary.json',{'status':'COMPLETE','source_sha256':frozen_source,'epoch':time.time()})
    except Exception as exc:
        atomic_json(a.output/'summary.json',{'status':'FAILED','reason':str(exc),'source_sha256':frozen_source,'epoch':time.time()})
        raise


if __name__=='__main__':main()
