"""Bounded GPU0 Flow-only FM control needed for Stage1 Shortcut comparison."""
from common import ROOT
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from fpsgen.hier_occ_flow.adaptation_gpu_slot import adaptation_admission
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('DESESCENE_GPU0_FLOW_CONTROL')!='1':raise ValueError('User-authorized GPU0 opt-in required')
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=source_sha();scripts=ROOT/'scripts/research_v2/hier_occ_flow'
    def run(name,script,args):
        while adaptation_admission(0)['status']!='AVAILABLE':time.sleep(15)
        if source_sha()!=frozen:raise RuntimeError('Registered source changed')
        status=a.output/(name+'.json')
        if status.exists():raise FileExistsError('Preserve prior complete or failed task; no silent retry')
        cmd=[sys.executable,'-u',str(scripts/script),*map(str,args)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',DESESCENE_SHARED_ADAPTATION='1',DESESCENE_GPU0_FLOW_CONTROL='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        with (a.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(status,dict(status='RUNNING',pid=child.pid,physical_gpu=0,command=cmd,source_sha256=frozen,epoch=time.time()))
            code=child.wait()
        atomic_json(status,dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,epoch=time.time()))
        if code:raise RuntimeError(name+' failed; no automatic retry/batch reduction')
    try:
        atomic_json(a.output/'supervisor.json',dict(status='ENGINEERING_THEN_FORMAL20K_THEN_EVAL900',pid=os.getpid(),source_sha256=frozen,epoch=time.time()))
        run('maximum_b2','verify_stage1_flow_control.py',['--config',a.config,'--output',a.output/'engineering',
            '--p0-root','/data-12/M2024-HWZ/FPSGen/worktrees/desescene-child-local-state-v1/experiments/desescene_child_local_state_v1/pilot_pair_v1',
            '--census','/data-12/M2024-HWZ/FPSGen/worktrees/desescene-state-condition-v1/experiments/desescene_state_condition_v1/phase0/d3_fixed_budget/per_frame.jsonl'])
        report=json.loads((a.output/'engineering/result.json').read_text())
        if report['status']!='PASS' or report['source_sha256']!=frozen:raise RuntimeError('Engineering evidence invalid')
        run('formal','train_encoder_adaptation.py',['--config',a.config,'--output',a.output/'run','--max-updates','10000'])
        value=json.loads((a.output/'run/summary.json').read_text())
        if value['successful_updates']!=10000 or value['successful_frame_exposures']!=20000:raise RuntimeError('Formal exposure budget mismatch')
        run('eval900','eval_encoder_adaptation.py',['--config',a.config,'--checkpoint',a.output/'run/checkpoints/resume_latest.pt',
            '--output',a.output/'evaluation/primary900.json'])
        atomic_json(a.output/'supervisor.json',dict(status='COMPLETE20K_AND_900',source_sha256=frozen,shortcut_training_started=False,epoch=time.time()))
    except Exception as exc:
        atomic_json(a.output/'supervisor.json',dict(status='FAILED_REVIEW_REQUIRED',reason=str(exc),source_sha256=frozen,epoch=time.time()))
        raise


if __name__=='__main__':main()
