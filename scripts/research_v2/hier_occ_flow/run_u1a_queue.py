"""Wait for baseline evaluations, then gate, smoke, train U1+A 20K and evaluate."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight

BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-u1-triplane-v1/experiments/desescene_u1_triplane_v1')
OUT=ROOT/'experiments/desescene_u1_abc_v1/u1a_queue'
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or OUT.exists():raise ValueError('GPU2 only; never overwrite experiment')
    OUT.mkdir(parents=True);env=os.environ.copy();env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None)
    env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    cfg=ROOT/'configs/research_v2/hier_occ_flow/patch_u1a_20k.yaml';gate=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';evaluate=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py'
    files=[cfg,gate,train,evaluate,Path(__file__),*[ROOT/'fpsgen/hier_occ_flow'/x for x in ('patch_unet_condition.py','gctv_encoder.py','stage1_network.py','stage1_occupancy.py','stage1_training.py')]];pins={str(x):file_sha(x) for x in files}
    def status(state,**extra):atomic_json(OUT/'status.json',dict(status=state,pid=os.getpid(),gpu=2,source_sha256=pins,**extra))
    def run(label,args):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned experiment source changed')
        with (OUT/(label+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            status('RUNNING_'+label,child_pid=child.pid)
            if child.wait():raise RuntimeError(label+' failed; preserve and inspect its log')
    try:
        status('WAITING_BASELINE_EVALUATIONS')
        while True:
            ss=[json.loads((BASE/f'continuation_{arm}/status.json').read_text()) for arm in ('p0','u1')]
            if any(x['status']=='BLOCKED' for x in ss):raise RuntimeError('A baseline is blocked; inspect it first')
            if all(x['status']=='COMPLETE_20K_AND_41' for x in ss):break
            time.sleep(20)
        status('WAITING_GPU2_ADMISSION')
        while gpu_preflight((2,),minimum_free_gib=22.,allow_shared_compute=False)['status']!='AVAILABLE':time.sleep(30)
        run('gate',[gate,'--config',cfg,'--output',OUT/'gate'])
        smoke=OUT/'smoke';base=[train,'--config',cfg,'--output',smoke,'--smoke']
        run('smoke2',base+['--max-updates','2']);run('smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
        s=json.loads((smoke/'summary.json').read_text());init=json.loads((smoke/'initialization_audit.json').read_text())
        if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34 or init['shared_flow_initialization_sha256']!='6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd':raise ValueError('Smoke or flow initialization failed')
        formal=OUT/'formal';run('train20k',[train,'--config',cfg,'--output',formal])
        summary=json.loads((formal/'summary.json').read_text())
        if summary['successful_updates']!=10000 or summary['successful_frame_exposures']!=20000:raise ValueError('20K not completed')
        run('seq08_nfe2',[evaluate,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',OUT/'seq08_41'])
        status('COMPLETE_20K_AND_41')
    except Exception as e:
        status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
