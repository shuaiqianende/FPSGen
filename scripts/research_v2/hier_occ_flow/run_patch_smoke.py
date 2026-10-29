"""Independent scratch 32-update P0/U1 smoke with exact resume."""
from pathlib import Path
import argparse,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('p0','u1'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    gpu={'p0':2,'u1':3}[a.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or a.output.exists():raise ValueError('Incorrect device or existing output')
    a.output.mkdir(parents=True);env=os.environ.copy();env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None)
    env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    cfg=ROOT/f'configs/research_v2/hier_occ_flow/patch_{a.arm}_20k.yaml';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';run=a.output/'run'
    files=[cfg,train,ROOT/'fpsgen/hier_occ_flow/patch_unet_condition.py',ROOT/'fpsgen/hier_occ_flow/stage1_training.py',ROOT/'fpsgen/hier_occ_flow/stage1_occupancy.py'];pins={str(x):file_sha(x) for x in files}
    try:
        for steps in (2,32):
            if any(file_sha(Path(k))!=v for k,v in pins.items()):raise ValueError('source changed during smoke')
            cmd=[sys.executable,'-u',str(train),'--config',str(cfg),'--output',str(run),'--smoke','--max-updates',str(steps)]
            if steps==32:cmd+=['--resume',str(run/'checkpoints/resume_latest.pt')]
            with (a.output/f'smoke_{steps}.log').open('x') as f:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                atomic_json(a.output/'status.json',dict(status=f'RUNNING_{steps}',pid=os.getpid(),child_pid=child.pid,arm=a.arm,gpu=gpu))
                if child.wait():raise RuntimeError(f'Smoke {steps} failed')
        s=json.loads((run/'summary.json').read_text());init=json.loads((run/'initialization_audit.json').read_text())
        if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34:raise ValueError('Smoke update accounting or AMP health failed')
        if init['shared_flow_initialization_sha256']!='6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd':raise ValueError('Flow initialization differs from registered baseline')
        atomic_json(a.output/'status.json',dict(status='PASS',arm=a.arm,gpu=gpu,summary=s,initialization=init,source_sha256=pins))
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='BLOCKED',arm=a.arm,gpu=gpu,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
