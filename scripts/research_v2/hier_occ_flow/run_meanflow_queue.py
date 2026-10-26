"""User-authorized two-arm MeanFlow queue, guarded by predecessor and GPU gates.

No new candidates, retries or reduced batches. A numerical replay exception must
pass the versioned strict bounds and its own original-FM control on the same GPU.
"""
from common import ROOT
import argparse,concurrent.futures,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import yaml
from fpsgen.hier_occ_flow.meanflow_queue import predecessor_status,project_jobs
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.meanflow_training import load_config
from fpsgen.hier_occ_flow.meanflow_control_audit import audit_control
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--once',action='store_true',help='Read-only scheduling decision; never launch');a=p.parse_args()
    plan=yaml.safe_load(a.queue.read_text());pin=source_sha();queue_sha=file_sha(a.queue)
    if plan['schema']!='desescene-meanflow-continuation-v1' or {t['id'] for t in plan['arms']}!={'S1-MF20K','S2-MF20K'}:
        raise ValueError('Only two registered independent MeanFlow experiments')
    if a.once:
        print(json.dumps(dict(source_sha256=pin,**predecessor_status(plan['predecessors'])),indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'supervisor.json').exists():raise FileExistsError('Existing queue identity/status requires explicit review; no blind restart')
    cpu=json.loads(Path(plan['cpu_report']).read_text())
    if cpu.get('status')!='CPU_PASS' or cpu['source_sha256']!=pin:raise ValueError('Own final source-matched CPU report required')
    config_pins={t['id']:file_sha(t['config']) for t in plan['arms']}
    def pinned():
        if source_sha()!=pin or file_sha(a.queue)!=queue_sha or any(file_sha(t['config'])!=config_pins[t['id']] for t in plan['arms']):
            raise RuntimeError('Pinned queue/source/config changed; no mutated code launch')
    def status(value):atomic_json(a.output/'supervisor.json',dict(pid=os.getpid(),source_sha256=pin,queue_sha256=queue_sha,epoch=time.time(),**value))
    try:
        while True:
            pinned();state=predecessor_status(plan['predecessors']);status(state)
            if state['status']=='BLOCKED_PREDECESSOR':return
            if state['status']=='READY':break
            time.sleep(15)
        def arm(task):
            ident=task['id'];gpu=task['physical_gpu'];root=Path(task['output']);root.mkdir(parents=True,exist_ok=True)
            card=(Path(plan['project_lock_root'])/f'meanflow_gpu{gpu}.lock');card.parent.mkdir(parents=True,exist_ok=True)
            gpu_lock=card.open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def mark(state,**extra):atomic_json(root/'queue_status.json',dict(arm=ident,status=state,physical_gpu=gpu,epoch=time.time(),**extra))
            def wait_slot():
                while True:
                    pinned();r=gpu_preflight((gpu,),minimum_free_gib=16.,allow_shared_compute=True)
                    jobs=project_jobs(gpu,r,plan['repository_root'])
                    if r['status']=='AVAILABLE' and not jobs:return r
                    mark('WAITING_GPU_ADMISSION',gpu_admission=r,existing_project_jobs=jobs);time.sleep(15)
            def run(name,script,args):
                wait_slot();pinned();progress=root/(name+'.task.json')
                if progress.exists():raise FileExistsError('Preserve previous task, including negative result: '+str(progress))
                cmd=[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow'/script),*map(str,args)]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),DESESCENE_MF_SHARED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    mark('RUNNING_'+name.upper(),pid=child.pid,command=cmd)
                    atomic_json(progress,dict(status='RUNNING',pid=child.pid,command=cmd,source_sha256=pin))
                    code=child.wait()
                atomic_json(progress,dict(status='COMPLETE' if code==0 else 'BLOCKED',exit_code=code,command=cmd,source_sha256=pin))
                return code
            try:
                cfg,g=load_config(task['config'])
                engineering=Path(cfg['engineering_report'])
                reuse=False
                if engineering.exists():
                    evidence=json.loads(engineering.read_text())
                    reuse=all(evidence.get(k)==v for k,v in dict(status='GPU_PASS',arm=ident,source_sha256=pin,config_sha256=config_sha(cfg)).items())
                    tested=[x['physical_gpu'] for x in evidence.get('gpu_admission',{}).get('gpus',[])]
                    if not reuse or tested!=[gpu]:raise ValueError('Existing engineering evidence is failed, mismatched, or from a different physical GPU')
                    atomic_json(root/'engineering_reuse.json',dict(status='VERIFIED_EXISTING',sha256=file_sha(engineering)))
                if not reuse and run('engineering','verify_meanflow_gpu.py',['--config',task['config'],'--output',root/'engineering']):
                    evidence=json.loads((root/'engineering/result.json').read_text()) if (root/'engineering/result.json').exists() else {}
                    mark(evidence.get('status','BLOCKED_ENGINEERING'),evidence=str(root/'engineering/result.json'),reason=evidence.get('reason'));return
                # An unmatched old checkpoint cannot silently become a control.
                control=audit_control(cfg);atomic_json(root/'control_audit.json',control)
                if control['status']!='MATCHED_CONTROL':mark('BLOCKED_CONTROL',audit=control);return
                smoke=Path(cfg['smoke_summary']);reuse_smoke=False
                if smoke.exists():
                    proof=json.loads(smoke.read_text());identity=proof.get('identity',{})
                    reuse_smoke=(proof.get('status')=='SMOKE_PASS' and proof.get('counts',{}).get('updates')==32
                        and proof['counts']['exposures']==64 and identity.get('source_sha256')==pin
                        and identity.get('config_sha256')==config_sha(cfg)
                        and identity.get('engineering_report_sha256')==file_sha(engineering))
                    if not reuse_smoke:raise ValueError('Existing Smoke evidence does not match source/config/engineering identity')
                    atomic_json(root/'smoke_reuse.json',dict(status='VERIFIED_EXISTING',sha256=file_sha(smoke)))
                if not reuse_smoke and run('smoke32','train_meanflow.py',['--config',task['config'],'--output',root/'smoke','--max-updates',32,'--smoke']):
                    mark('BLOCKED_SMOKE');return
                for updates in (100,500,10000):
                    args=['--config',task['config'],'--output',root/'formal','--max-updates',updates]
                    if updates!=100:args+=['--resume',root/'formal/checkpoints/resume_latest.pt']
                    if run(f'formal_{updates}','train_meanflow.py',args):mark('BLOCKED_TRAINING',at_updates=updates);return
                    proof=json.loads((root/'formal/summary.json').read_text())
                    if proof['counts']['updates']!=updates or proof['counts']['exposures']!=2*updates:raise ValueError('Successful budget mismatch')
                for split in ('development900','seq08_41'):
                    if run(split,'eval_meanflow.py',['--config',task['config'],'--checkpoint',root/'formal/checkpoints/resume_latest.pt',
                        '--output',root/'evaluation'/split,'--split',split]):mark('BLOCKED_EVALUATION',split=split);return
                # Quality has finished. Keep only a low-priority CPU waiter
                # for uncontended timing; never occupy the GPU while waiting.
                while True:
                    pinned();exclusive=gpu_preflight((gpu,),minimum_free_gib=16.,allow_shared_compute=False)
                    if exclusive['status']=='AVAILABLE':break
                    mark('EVALUATED_WAITING_EXCLUSIVE_TIMING',formal_exposures=20000,development_frames=900,seq08_frames=41,gpu_admission=exclusive)
                    time.sleep(15)
                if run('paired_timing','benchmark_meanflow.py',['--config',task['config'],'--checkpoint',root/'formal/checkpoints/resume_latest.pt',
                    '--output',root/'benchmark']):mark('EVALUATED_TIMING_FAILED');return
                benchmark=json.loads((root/'benchmark/status.json').read_text())
                mark('EVALUATED',formal_exposures=20000,development_frames=900,seq08_frames=41,benchmark=benchmark)
            except Exception as exc:
                mark('BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        status(dict(status='PREDECESSORS_COMPLETE_ENGINEERING_THEN_REGISTERED_ARMS',arms=plan['arms']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(arm,plan['arms']))
        states={t['id']:json.loads((Path(t['output'])/'queue_status.json').read_text()) for t in plan['arms']}
        status(dict(status='QUEUE_FINISHED_REVIEW_RESULTS',arms=states,automatic_new_training=False))
    except Exception as exc:
        status(dict(status='BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
