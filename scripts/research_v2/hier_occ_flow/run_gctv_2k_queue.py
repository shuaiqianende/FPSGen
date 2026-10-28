"""After 41-frame D0, run own gate, independent smoke and 2K exposures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--dependency',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or any(x not in ('g0','g1','g2') for x in a.models):raise ValueError('Registered device/models required')
    if a.root.exists():raise FileExistsError('Preserve prior results')
    a.root.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    source=[ROOT/'fpsgen/hier_occ_flow'/x for x in ('stage1_occupancy.py','stage1_training.py','gctv_encoder.py','triplane_encoder.py')];pin={str(x):file_sha(x) for x in source}
    def run(label,command):
        if any(file_sha(Path(k))!=v for k,v in pin.items()):raise RuntimeError('Source changed after registration')
        with (a.root/(label+'.log')).open('x') as f:
            c=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            atomic_json(a.root/'status.json',dict(status='RUNNING_'+label,pid=os.getpid(),child_pid=c.pid,gpu=a.gpu,command=command,source_sha256=pin))
            if c.wait():raise RuntimeError(label+' failed; no automatic retry')
    try:
        atomic_json(a.root/'status.json',dict(status='WAITING_FOR_41_FRAME_D0',gpu=a.gpu,pid=os.getpid(),dependency=str(a.dependency)))
        while True:
            if a.dependency.exists():
                st=json.loads(a.dependency.read_text())['status']
                if st=='COMPLETE':break
                if st=='BLOCKED':raise RuntimeError('D0 dependency failed')
            time.sleep(20)
        for name in a.models:
            cfg=ROOT/f'configs/research_v2/hier_occ_flow/gctv_{name}_20k.yaml';out=a.root/name;out.mkdir();gate=out/'gate'
            run(name+'_gate',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py'),'--config',str(cfg),'--output',str(gate)])
            smoke=out/'smoke';trainer=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py'
            base=[sys.executable,'-u',str(trainer),'--config',str(cfg),'--output',str(smoke),'--smoke']
            run(name+'_smoke2',base+['--max-updates','2']);run(name+'_smoke32',base+['--max-updates','32','--resume',str(smoke/'checkpoints/resume_latest.pt')])
            s=json.loads((smoke/'summary.json').read_text())
            if s['successful_updates']!=32 or s['successful_frame_exposures']!=64:raise ValueError('Smoke incomplete')
            formal=out/'formal';run(name+'_train2k',[sys.executable,'-u',str(trainer),'--config',str(cfg),'--output',str(formal),'--max-updates','1000'])
            s=json.loads((formal/'summary.json').read_text())
            if s['successful_updates']!=1000 or s['successful_frame_exposures']!=2000:raise ValueError('2K exposure checkpoint incomplete')
            run(name+'_eval41',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/eval_triplane_seq08.py'),'--config',str(cfg),'--checkpoint',str(formal/'checkpoints/resume_latest.pt'),'--output',str(out/'evaluation_2k_seq08_41')])
        atomic_json(a.root/'status.json',dict(status='COMPLETE_2K_AND_41',models=a.models,gpu=a.gpu,pid=os.getpid(),automatic_5k=False))
    except Exception as e:
        atomic_json(a.root/'status.json',dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),gpu=a.gpu,pid=os.getpid()));raise

if __name__=='__main__':main()
