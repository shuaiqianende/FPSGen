"""Wait for C1, evaluate/diagnose and run 32-step smokes; never starts formal training."""
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
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

PYTHON=Path('/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-worktree',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=a.source_worktree.resolve()
    runs=base/'experiments/desescene_v_next'
    configs=ROOT/'configs/research_v2/hier_occ_flow'
    scripts=ROOT/'scripts/research_v2/hier_occ_flow'
    mature=Path('/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes')
    jobs=[]

    def run(name,gpu,script,args,completion):
        completion=Path(completion)
        state_path=a.output/(name+'.status.json')
        if completion.exists():
            status=json.loads(completion.read_text()).get('status')
            if status in ('BLOCKED','FAIL'):raise RuntimeError('Prior failed output: '+str(completion))
            print('Existing completion retained',name,flush=True);return
        while gpu_preflight((gpu,))['status']!='AVAILABLE':
            atomic_json(state_path,{'status':'WAITING_GPU_IDLE','physical_gpu':gpu,'epoch':time.time()})
            time.sleep(20)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='2',TMPDIR='/data-12/M2024-HWZ/FPSGen/env/tmp',
                 HS_STAGE2_ALLOW_SHARED_COMPUTE='0',
                 TORCH_EXTENSIONS_DIR='/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/torch_extensions')
        cmd=[('bash' if Path(script).suffix=='.sh' else str(PYTHON)),str(script),*map(str,args)]
        atomic_json(state_path,{'status':'RUNNING','command':cmd,'physical_gpu':gpu,'epoch':time.time()})
        print('Starting',name,'GPU',gpu,flush=True)
        with (a.output/(name+'.log')).open('a',buffering=1) as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        atomic_json(state_path,{'status':'COMPLETE' if code==0 and completion.exists() else 'FAILED',
                               'command':cmd,'exit_code':code,'epoch':time.time()})
        if code or not completion.exists():raise RuntimeError(name+' failed; see log')
        output=json.loads(completion.read_text())
        if output.get('status') in ('BLOCKED','FAIL'):raise RuntimeError(name+' did not pass')
        print('Completed',name,flush=True)

    def wait_c1(name,gpu,nfe):
        folder=runs/name
        while True:
            summary=folder/'summary.json'
            if summary.exists():
                x=json.loads(summary.read_text())
                if x.get('successful_frame_exposures')==20000 and x.get('successful_updates')==10000:
                    break
            launch=folder/'launch.json'
            if launch.exists():
                pid=json.loads(launch.read_text())['rank0_pid']
                try:os.kill(pid,0)
                except ProcessLookupError:raise RuntimeError(name+' stopped before registered budget; no automatic restart')
            atomic_json(a.output/('wait_'+name+'.json'),{'status':'WAITING_EXISTING_TRAINING','physical_gpu':gpu,'epoch':time.time()})
            time.sleep(20)
        output=a.output/'c1'/(name+'_900.json')
        run('eval_'+name,gpu,scripts/'eval_stage2_384.py',[
            '--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt',
            '--output',output,'--ema','--nfes',nfe,'--max-frames',900,
            '--sampling-seeds',20261031,'--continuous-metrics'],output)

    def mechanisms(name,gpu,folder,checkpoint,ema=False):
        output=a.output/name
        args=['--config',folder/'config.json','--checkpoint',checkpoint,'--output',output,
              '--max-frames',64,'--mode','both']
        if ema:args+=['--ema']
        run(name,gpu,scripts/'audit_desescene_mechanisms.py',args,output/'summary.json')

    def gpu2():
        wait_c1('static_context_fm_20k',2,4)
        mechanisms('d1_d2_frozen_dynamic',2,mature,mature/'checkpoints/step025000.pt')

    def gpu3():
        wait_c1('static_context_bce_20k',3,1)
        ranking=a.output/'d3_fixed_budget'
        run('d3_fixed_budget',3,scripts/'audit_desescene_fixed_budget.py',[
            '--config',configs/'stage2_384_desescene_sci_b0_20k.yaml','--output',ranking,'--max-frames',900],ranking/'summary.json')
        verify=a.output/'m1_gpu_verification'
        run('m1_gpu_verification',3,scripts/'verify_desescene_m1_gpu.py',[
            '--b0-config',configs/'stage2_384_desescene_sci_b0_20k.yaml',
            '--m1-config',configs/'stage2_384_desescene_sci_m1_20k.yaml',
            '--stress-source',ranking/'per_frame.jsonl','--output',verify],verify/'result.json')
        gate_path=ROOT/'experiments/desescene_state_condition_v1/gate.json'
        if not gate_path.exists():
            base_gate=runs/'engineering_gate_child_bridge.json'
            gate=json.loads(base_gate.read_text())
            gate.update(research_model_smoke=False,state_condition_gpu_verified=False,c1_review_complete=False,
                        inherited_engineering_gate_sha256=file_sha(base_gate))
            atomic_json(gate_path,gate)
        smokes=[]
        for arm in ('b0','m1'):
            output=a.output/('smoke_'+arm)
            run('smoke_'+arm,3,scripts/'launch_stage2_384.sh',[
                '--config',configs/('stage2_384_desescene_sci_'+arm+'_smoke.yaml'),
                '--output',output,'--max-updates',32],output/'summary.json')
            smokes.append(output)
        subprocess.run([str(PYTHON),str(scripts/'register_desescene_m1_gate.py'),
                        '--gate',str(gate_path),'--gpu-verification',str(verify/'result.json'),
                        '--smoke-outputs',*map(str,smokes)],cwd=ROOT,check=True)
        order=('r0_gpu2_20k','e1_gpu3_20k','e2_gpu2_20k','e3_gpu3_20k')
        for round_id in range(3):
            for name in (order if round_id!=1 else tuple(reversed(order))):
                folder=base/'experiments/dense_fidelity_v2_single_gpu'/name
                output=a.output/'d4_encoder'/(name+'_round'+str(round_id+1)+'.json')
                run('d4_'+name+'_round'+str(round_id+1),3,scripts/'benchmark_trained_encoder_pilot.py',[
                    '--config',folder/'config.json','--checkpoint',folder/'checkpoints/step010000.pt',
                    '--output',output,'--physical-gpu',3,'--rounds',1,'--precision','explicit_fp16'],output)
        for name in ('static_context_fm_20k','static_context_bce_20k'):
            folder=runs/name
            mechanisms('d1_d2_'+name,3,folder,folder/'checkpoints/resume_latest.pt',ema=True)

    atomic_json(a.output/'queue_contract.json',{'source_worktree':str(base),'new_worktree':str(ROOT),'supervisor_pid':os.getpid(),
        'tasks':['C1 primary900 raw-map geometry','D1/D2 64 scenes','D3 fixed-budget900',
                 'M1 real GPU correctness','B0/M1 engineering32-step smokes','D4 trained Encoder paired5+20 x3'],
        'new_formal_training_started_by_this_queue':False,'engineering_smoke_updates_per_arm':32,
        'm1_training_requires_research_review':True})
    failures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(gpu2),pool.submit(gpu3)]
        for future in futures:
            try:future.result()
            except Exception as exc:failures.append(str(exc))
    if not failures:
        subprocess.run([str(PYTHON),str(scripts/'summarize_desescene_phase0.py'),
                        '--phase0-root',str(a.output)],cwd=ROOT,check=True)
        subprocess.run([str(PYTHON),str(scripts/'audit_desescene_evidence.py'),
                        '--source-worktree',str(base),'--output',str(a.output/'final_evidence'),
                        '--c1-evaluation-root',str(a.output/'c1')],cwd=ROOT,check=True)
    atomic_json(a.output/'summary.json',{'status':'FAILED' if failures else 'DIAGNOSTICS_COMPLETE_REVIEW_REQUIRED',
                'failures':failures,'new_formal_training_started':False,'epoch':time.time()})
    if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':
    main()
