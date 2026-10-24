"""Registered 20K local-state pair: real GPU gate, smokes, staged training and900 eval."""
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
from fpsgen.hier_occ_flow.stage2_runtime import source_sha,load_stage2_config,config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from run_desescene_m1_matched_queue import health_report
from register_desescene_m1_gate import validate_smoke

PYTHON='/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python'
SCRIPTS=ROOT/'scripts/research_v2/hier_occ_flow'
CONFIGS=ROOT/'configs/research_v2/hier_occ_flow'


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-verification',type=Path,required=True)
    p.add_argument('--verification-pid',type=int)
    p.add_argument('--inherited-gate',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen_source=source_sha();gpu_map={'position':1,'attention':3}
    atomic_json(a.output/'contract.json',{'status':'REGISTERED','supervisor_pid':os.getpid(),
        'source_sha256':frozen_source,'physical_gpus':gpu_map,'training_seed':20261032,
        'successful_frame_exposures_per_arm':20000,'batch_per_gpu':2,'staged_updates':[100,500,10000],
        'smoke_gpu':3,'encoder_followups':'Registered separately; not yet engineering-ready',
        'no_automatic_100k':True,'epoch':time.time()})
    def run(name,gpu,script,args,completion,expected=None):
        if source_sha()!=frozen_source:raise RuntimeError('Source changed; stop before the next task')
        if completion.exists():
            value=json.loads(completion.read_text())
            if value.get('status') in ('FAILED','FAIL','BLOCKED'):raise RuntimeError('Retained failed output '+str(completion))
            if expected is None or value.get('successful_updates',0)>=expected:return
        status=a.output/(name+'.status.json')
        while gpu_preflight((gpu,))['status']!='AVAILABLE':
            atomic_json(status,{'status':'WAITING_GPU_IDLE','physical_gpu':gpu,'epoch':time.time()})
            time.sleep(20)
        cmd=[('bash' if script.suffix=='.sh' else PYTHON),str(script),*map(str,args)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='2',HS_STAGE2_ALLOW_SHARED_COMPUTE='0',
                 TMPDIR='/data-12/M2024-HWZ/FPSGen/env/tmp',
                 TORCH_EXTENSIONS_DIR='/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/torch_extensions')
        with (a.output/(name+'.log')).open('a',buffering=1) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(status,{'status':'RUNNING','pid':child.pid,'physical_gpu':gpu,'command':cmd,'epoch':time.time()})
            code=child.wait()
        valid=code==0 and completion.exists()
        atomic_json(status,{'status':'COMPLETE' if valid else 'FAILED','exit_code':code,'epoch':time.time()})
        if not valid:raise RuntimeError(name+' failed; inspect preserved log')
        if json.loads(completion.read_text()).get('status') in ('FAIL','FAILED','BLOCKED'):
            raise RuntimeError(name+' did not pass')
    try:
        while not a.gpu_verification.exists():
            if a.verification_pid:
                try:os.kill(a.verification_pid,0)
                except ProcessLookupError:raise RuntimeError('GPU verification exited without a report')
            atomic_json(a.output/'status.json',{'status':'WAITING_GPU_VERIFICATION','epoch':time.time()})
            time.sleep(5)
        verified=json.loads(a.gpu_verification.read_text())
        if verified['status']!='PASS' or verified['source_sha256']!=frozen_source:
            raise RuntimeError('Actual GPU correctness/source identity not verified')
        gate=json.loads(a.inherited_gate.read_text())
        gate.update(research_model_smoke=False,local_state_gpu_verified=False,
                    local_state_verified_modes=[],source_sha256=frozen_source,
                    initialization_checkpoint_sha256=verified['source_initialization']['checkpoint_sha256'])
        gate_path=ROOT/'experiments/desescene_child_local_state_v1/gate.json'
        atomic_json(gate_path,gate);smoke_evidence=[]
        for arm in gpu_map:
            cfg,g=load_stage2_config(CONFIGS/f'stage2_384_child_local_{arm}_20k.yaml')
            record=verified['variants'][cfg['model']['local_state_encoding']]
            if record['config_sha256']!=config_sha(cfg):raise RuntimeError('GPU test config differs')
            folder=a.output/('engineering_'+arm)
            run('engineering_'+arm,3,SCRIPTS/'launch_stage2_384.sh',[
                '--config',CONFIGS/f'stage2_384_child_local_{arm}_smoke.yaml','--output',folder,
                '--max-updates',32],folder/'summary.json',32)
            rows=[json.loads(x) for x in (folder/'train_rank0.jsonl').read_text().splitlines() if x.strip()]
            smoke_evidence.append({'arm':arm,**validate_smoke(json.loads((folder/'summary.json').read_text()),
                json.loads((folder/'launch.json').read_text()),json.loads((folder/'config.json').read_text()),rows,frozen_source)})
        gate.update(research_model_smoke=True,local_state_gpu_verified=True,
                    local_state_verified_modes=list(verified['variants']),smoke_evidence=smoke_evidence,
                    local_state_gpu_evidence_sha256=file_sha(a.gpu_verification))
        atomic_json(gate_path,gate)
        def train(arm):
            gpu=gpu_map[arm];folder=a.output/(arm+'_20k')
            for stop in (100,500,10000):
                args=['--config',CONFIGS/f'stage2_384_child_local_{arm}_20k.yaml','--output',folder,'--max-updates',stop]
                if (folder/'checkpoints/resume_latest.pt').exists():args+=['--resume',folder/'checkpoints/resume_latest.pt']
                run(f'{arm}_train_{stop}',gpu,SCRIPTS/'launch_stage2_384.sh',args,folder/'summary.json',stop)
                health_report(folder,stop)
            evaluation=folder/'evaluation/primary900_nfe4.json'
            run(arm+'_eval900',gpu,SCRIPTS/'eval_stage2_384.py',[
                '--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt',
                '--output',evaluation,'--ema','--max-frames',900,'--nfes',4,'--sampling-seeds',20261031,
                '--continuous-metrics'],evaluation)
            mechanisms=folder/'evaluation/mechanisms64'
            run(arm+'_mechanisms64',gpu,SCRIPTS/'audit_desescene_mechanisms.py',[
                '--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt',
                '--output',mechanisms,'--ema','--max-frames',64,'--mode','both'],mechanisms/'summary.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train,arm) for arm in gpu_map]
            failures=[]
            for future in futures:
                try:future.result()
                except Exception as exc:failures.append(str(exc))
            if failures:raise RuntimeError('; '.join(failures))
        atomic_json(a.output/'status.json',{'status':'COMPLETE','epoch':time.time()})
    except Exception as exc:
        atomic_json(a.output/'status.json',{'status':'FAILED','reason':str(exc),'epoch':time.time()})
        raise


if __name__=='__main__':main()
