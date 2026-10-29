"""Resume the verified 5K scratch pilot through 20K then run seq08 NFE2."""
from pathlib import Path
import argparse,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('p0','u1'),required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();gpu={'p0':2,'u1':3}[a.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or a.output.exists():raise ValueError('Incorrect GPU or output already exists')
    s=json.loads((a.pilot/'status.json').read_text())
    if s['status']!='COMPLETE_5K_AWAITING_SCREEN' or s['arm']!=a.arm or any(file_sha(Path(k))!=v for k,v in s['source_sha256'].items()):raise ValueError('5K pilot or source identity mismatch')
    for phase in ('dev_2k','dev_5k'):
        d=json.loads((a.pilot/phase/'results.json').read_text())
        if d['status']!='COMPLETE_DEV41' or len(d['result']['nfe_results']['2']['rows'])!=41 or any(r['sequence']=='08' for r in d['result']['nfe_results']['2']['rows']):raise ValueError('Expected independent dev41')
    a.output.mkdir(parents=True);env=os.environ.copy();env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None);env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    cfg=ROOT/f'configs/research_v2/hier_occ_flow/patch_{a.arm}_20k.yaml';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py';formal=a.pilot/'formal';ck=formal/'checkpoints/resume_latest.pt'
    pins=dict(s['source_sha256']);pins[str(ev)]=file_sha(ev);pins[str(Path(__file__))]=file_sha(Path(__file__))
    def run(name,args):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Source identity changed')
        with (a.output/(name+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            atomic_json(a.output/'status.json',dict(status='RUNNING_'+name,arm=a.arm,gpu=gpu,pid=os.getpid(),child_pid=child.pid))
            if child.wait():raise RuntimeError(name+' failed; see log')
    try:
        run('train20k',[train,'--config',cfg,'--output',formal,'--resume',ck]);summary=json.loads((formal/'summary.json').read_text())
        if summary['successful_updates']!=10000 or summary['successful_frame_exposures']!=20000:raise ValueError('20K budget not completed')
        run('seq08_nfe2',[ev,'--config',cfg,'--checkpoint',ck,'--output',a.output/'seq08_41'])
        atomic_json(a.output/'status.json',dict(status='COMPLETE_20K_AND_41',arm=a.arm,gpu=gpu))
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='BLOCKED',arm=a.arm,gpu=gpu,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
