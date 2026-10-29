"""Train P0 or U1 from scratch through 5K, with fixed development screens."""
from pathlib import Path
import argparse,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('p0','u1'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True);a=p.parse_args();gpu={'p0':2,'u1':3}[a.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or a.output.exists():raise ValueError('Incorrect GPU or existing output')
    s=json.loads((a.smoke/'status.json').read_text());
    if s.get('status')!='PASS' or s.get('arm')!=a.arm or any(file_sha(Path(k))!=v for k,v in s['source_sha256'].items()):raise ValueError('Fresh matching smoke required')
    a.output.mkdir(parents=True);env=os.environ.copy();env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None)
    env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    cfg=ROOT/f'configs/research_v2/hier_occ_flow/patch_{a.arm}_20k.yaml';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';dev=ROOT/'scripts/research_v2/hier_occ_flow/eval_highres_dev41.py';formal=a.output/'formal'
    files=[Path(k) for k in s['source_sha256']]+[dev,Path(__file__)];pins={str(x):file_sha(x) for x in files}
    def run(label,args):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Source changed while running')
        with (a.output/(label+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            atomic_json(a.output/'status.json',dict(status='RUNNING_'+label,pid=os.getpid(),child_pid=child.pid,arm=a.arm,gpu=gpu))
            if child.wait():raise RuntimeError(label+' failed')
    try:
        run('train2k',[train,'--config',cfg,'--output',formal,'--max-updates','1000'])
        ck=formal/'checkpoints/resume_latest.pt'
        run('dev2k',[dev,'--config',cfg,'--checkpoint',ck,'--output',a.output/'dev_2k','--gpu',gpu])
        run('train5k',[train,'--config',cfg,'--output',formal,'--max-updates','2500','--resume',ck])
        run('dev5k',[dev,'--config',cfg,'--checkpoint',ck,'--output',a.output/'dev_5k','--gpu',gpu])
        atomic_json(a.output/'status.json',dict(status='COMPLETE_5K_AWAITING_SCREEN',arm=a.arm,gpu=gpu,source_sha256=pins))
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='BLOCKED',arm=a.arm,gpu=gpu,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
