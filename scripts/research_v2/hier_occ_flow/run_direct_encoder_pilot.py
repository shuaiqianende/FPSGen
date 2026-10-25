"""User-directed EF1/EF2 formal20K; keep failed replay evidence and strict counters."""
from common import ROOT
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fpsgen.hier_occ_flow.adaptation_gpu_slot import adaptation_admission
from fpsgen.hier_occ_flow.storage import atomic_json
from fpsgen.hier_occ_flow.stage2_runtime import source_sha


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    policy=json.loads(args.policy.read_text());frozen=source_sha()
    if frozen!=policy['execution_source_sha256']:raise ValueError('Source policy mismatch')
    lock=(ROOT/'experiments/desescene_encoder_adaptation_v1/supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    legacy=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-encoder-adaptation-v1/experiments/desescene_encoder_adaptation_v1/supervisor.lock').open('a')
    fcntl.flock(legacy,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scripts=ROOT/'scripts/research_v2/hier_occ_flow'

    def execute(arm,gpu,*,reference=False):
        while adaptation_admission(gpu)['status']!='AVAILABLE':time.sleep(15)
        if source_sha()!=frozen:raise RuntimeError('Source changed before starting a new task')
        cfg=ROOT/f'configs/research_v2/hier_occ_flow/encoder_adaptation_{arm.lower()}_20k.yaml'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),DESESCENE_SHARED_ADAPTATION='1',
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
        def launch(name,command):
            status=args.output/(name+'.json');log=args.output/(name+'.log')
            if status.exists():raise FileExistsError('Preserve existing task; review exact resume explicitly')
            with log.open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                atomic_json(status,dict(status='RUNNING',pid=child.pid,physical_gpu=gpu,
                    source_sha256=frozen,command=command,epoch=time.time(),
                    strict_replay_certification='NOT_CERTIFIED_NUMERICAL_REPLAY'))
                code=child.wait()
            atomic_json(status,dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,
                physical_gpu=gpu,source_sha256=frozen,epoch=time.time()))
            if code:raise RuntimeError(name+' failed; no automatic retry or budget change')
        if not reference:
            output=args.output/'runs'/arm
            launch(arm+'_formal',[sys.executable,'-u',str(scripts/'train_encoder_adaptation.py'),
                '--config',str(cfg),'--output',str(output),'--max-updates','10000',
                '--numerical-pilot-policy',str(args.policy)])
            summary=json.loads((output/'summary.json').read_text())
            if summary['successful_updates']!=10000 or summary['successful_frame_exposures']!=20000:
                raise RuntimeError('Formal successful exposure budget mismatch')
        eval_cfg=ROOT/f'configs/research_v2/hier_occ_flow/encoder_adaptation_{"ef1" if reference else arm.lower()}_20k.yaml'
        command=[sys.executable,'-u',str(scripts/'eval_encoder_adaptation.py'),
                 '--config',str(eval_cfg),'--output',str(args.output/'evaluations'/f'{arm}.json')]
        command+=['--reference'] if reference else ['--checkpoint',str(args.output/'runs'/arm/'checkpoints/resume_latest.pt')]
        launch(arm+'_eval900',command)

    atomic_json(args.output/'supervisor.json',dict(status='FORMAL_EF1_EF2_LAUNCHING',pid=os.getpid(),
        source_sha256=frozen,seed=20261032,true_batch=2,exposures_per_arm=20000,
        strict_replay_certification='NOT_CERTIFIED_NUMERICAL_REPLAY',epoch=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(execute,arm,gpu) for arm,gpu in [('EF1',2),('EF2',3)]]
            errors=[]
            for job in jobs:
                try:job.result()
                except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
        if errors:raise RuntimeError('; '.join(errors))
        execute('EF0',2,reference=True)
        atomic_json(args.output/'supervisor.json',dict(status='P1_FORMAL_AND_QUALITY_COMPLETE',
            source_sha256=frozen,p2_started=False,p3_started=False,formal_timing='DEFERRED',epoch=time.time()))
    except Exception as exc:
        atomic_json(args.output/'supervisor.json',dict(status='FAILED_REVIEW_REQUIRED',reason=str(exc),epoch=time.time()))
        raise


if __name__=='__main__':main()
