"""Resume GCTV runs from their own 5K checkpoint to20K exposures and evaluate41."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--dependency',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or any(m not in ('g0','g1','g2') for m in a.models):raise ValueError('Registered GPU/models required')
    if a.root.exists():raise FileExistsError('Preserve existing run')
    a.root.mkdir(parents=True);base=a.dependency.parent.parent/f'2k_queue_gpu{a.gpu}'
    env=os.environ.copy();env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    source=[ROOT/'fpsgen/hier_occ_flow'/x for x in ('stage1_occupancy.py','stage1_training.py','gctv_encoder.py','triplane_encoder.py')];pin={str(p):file_sha(p) for p in source}
    try:
        atomic_json(a.root/'status.json',dict(status='WAITING_FOR_5K_AND_41',gpu=a.gpu,pid=os.getpid(),dependency=str(a.dependency)))
        while True:
            if a.dependency.exists():
                status=json.loads(a.dependency.read_text())['status']
                if status=='COMPLETE_5K_AND_41':break
                if status=='BLOCKED':raise RuntimeError('5K queue blocked; inspect prior logs')
            time.sleep(20)
        for name in a.models:
            if any(file_sha(Path(k))!=v for k,v in pin.items()):raise ValueError('Source changed before continuation')
            model=base/name;formal=model/'formal';summary=json.loads((formal/'summary.json').read_text());quality=json.loads((model/'evaluation_5k_seq08_41/results.json').read_text())
            if summary['successful_updates']!=2500 or summary['successful_frame_exposures']!=5000 or quality['status']!='EVALUATED_SEQ08_41':raise ValueError('Own5K run and 41-frame evaluation required')
            cfg=ROOT/f'configs/research_v2/hier_occ_flow/gctv_{name}_20k.yaml';checkpoint=formal/'checkpoints/resume_latest.pt'
            commands=[('train20k',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py'),'--config',str(cfg),'--output',str(formal),'--resume',str(checkpoint)]),('eval20k',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/eval_triplane_seq08.py'),'--config',str(cfg),'--checkpoint',str(checkpoint),'--output',str(model/'evaluation_20k_seq08_41')])]
            for phase,command in commands:
                with (a.root/(name+'_'+phase+'.log')).open('x') as f:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                    atomic_json(a.root/'status.json',dict(status='RUNNING_'+name+'_'+phase,gpu=a.gpu,pid=os.getpid(),child_pid=child.pid,checkpoint_sha256=file_sha(checkpoint),command=command,source_sha256=pin))
                    if child.wait():raise RuntimeError(name+' '+phase+' failed; preserve logs, no retry')
                if phase=='train20k':
                    s=json.loads((formal/'summary.json').read_text())
                    if s['successful_updates']!=10000 or s['successful_frame_exposures']!=20000:raise ValueError('20K exposure target incomplete')
                else:
                    s=json.loads((model/'evaluation_20k_seq08_41/results.json').read_text())
                    if s['status']!='EVALUATED_SEQ08_41':raise ValueError('41-frame evaluation incomplete')
        atomic_json(a.root/'status.json',dict(status='COMPLETE_20K_AND_41',gpu=a.gpu,pid=os.getpid(),models=a.models))
    except Exception as e:
        atomic_json(a.root/'status.json',dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),gpu=a.gpu,pid=os.getpid()));raise

if __name__=='__main__':main()
