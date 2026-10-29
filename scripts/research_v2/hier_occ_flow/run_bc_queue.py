"""Per-device sequential 20K training; reject failed gates, never overwrite."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or a.output.exists():raise ValueError('Invalid GPU or existing queue')
    a.output.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    arms=('u1ab','u1abctl') if a.gpu==2 else ('u1ac','u1abc')
    files=[Path(__file__),*[ROOT/'fpsgen/hier_occ_flow'/x for x in ('patch_unet_condition.py','stage1_occupancy.py','stage1_training.py','gctv_encoder.py')],*[ROOT/'scripts/research_v2/hier_occ_flow'/x for x in ('verify_triplane_gpu.py','train_dense_fidelity_single_gpu.py','eval_patch_seq08_nfe2.py')],*[ROOT/f'configs/research_v2/hier_occ_flow/patch_{name}_20k.yaml' for name in arms]];pins={str(f):file_sha(f) for f in files}
    def status(state,**extra):atomic_json(a.output/'status.json',dict(status=state,pid=os.getpid(),gpu=a.gpu,source_sha256=pins,**extra))
    def run(name,args):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned source changed')
        with (a.output/(name+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+name,child_pid=child.pid)
            if child.wait():raise RuntimeError(name+' failed; see log')
    try:
        results={}
        for arm in arms:
            status('WAITING_GPU_ADMISSION',arm=arm)
            while gpu_preflight((a.gpu,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
            out=a.output/arm;out.mkdir();cfg=ROOT/f'configs/research_v2/hier_occ_flow/patch_{arm}_20k.yaml';gate=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py'
            try:
                run(arm+'_gate',[gate,'--config',cfg,'--output',out/'gate'])
                smoke=out/'smoke';base=[train,'--config',cfg,'--output',smoke,'--smoke'];run(arm+'_smoke2',base+['--max-updates','2']);run(arm+'_smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
                s=json.loads((smoke/'summary.json').read_text());i=json.loads((smoke/'initialization_audit.json').read_text())
                if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34 or i['shared_flow_initialization_sha256']!='6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd':raise ValueError('Smoke or Flow identity failed')
                run(arm+'_train20k',[train,'--config',cfg,'--output',out/'formal'])
                s=json.loads((out/'formal/summary.json').read_text())
                if s['successful_updates']!=10000 or s['successful_frame_exposures']!=20000:raise ValueError('20K not completed')
                run(arm+'_seq08_41',[ev,'--config',cfg,'--checkpoint',out/'formal/checkpoints/resume_latest.pt','--output',out/'seq08_41'])
                results[arm]='COMPLETE_20K_AND_41'
            except Exception as e:
                results[arm]='BLOCKED';atomic_json(out/'failure.json',dict(reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()))
            atomic_json(a.output/'progress.json',dict(results=results))
        status('COMPLETE',results=results)
    except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
