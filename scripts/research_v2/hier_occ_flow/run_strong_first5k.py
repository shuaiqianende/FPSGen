"""Run one registered S1/S5 arm through independent 5K evaluation only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse, json, os, subprocess, traceback
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha

CONTROL=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-gctv-matched-fm-control')
DATA_AUDIT='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('s1','s5'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    gpu={'s1':2,'s5':3}[a.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('physical GPU mismatch')
    if a.output.exists():raise FileExistsError('Existing results are never overwritten')
    a.output.mkdir(parents=True); env=os.environ.copy();env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']=DATA_AUDIT;env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None)
    cfg=ROOT/f'configs/research_v2/hier_occ_flow/strong_{a.arm}_20k.yaml'
    train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';dev=ROOT/'scripts/research_v2/hier_occ_flow/eval_highres_dev41.py'
    files=[cfg,train,dev,Path(__file__),*[ROOT/'fpsgen/hier_occ_flow'/x for x in ('stage1_network.py','stage1_occupancy.py','stage1_training.py','strong_encoder_v2.py')]]
    pins={str(f):file_sha(f) for f in files}
    def run(label,args):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned source changed')
        with (a.output/(label+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            atomic_json(a.output/'status.json',dict(status='RUNNING_'+label,pid=os.getpid(),child_pid=child.pid,arm=a.arm,gpu=gpu,source_sha256=pins))
            if child.wait():raise RuntimeError(label+' failed; inspect log')
    try:
        smoke=a.output/'smoke'; formal=a.output/'formal'
        base=[train,'--config',cfg,'--output',smoke,'--smoke']
        run('smoke2',base+['--max-updates','2']);run('smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
        s=json.loads((smoke/'summary.json').read_text()); init=json.loads((smoke/'initialization_audit.json').read_text())
        ref=json.loads((CONTROL/'experiments/desescene_gctv_matched_control/run/formal/initialization_audit.json').read_text())
        if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34:raise ValueError('Smoke did not pass numerical health checks')
        if init['shared_flow_initialization_sha256']!=ref['shared_flow_initialization_sha256'] or init['matched_pair_initialization']['reference_r0_template_sha256']!=ref['matched_pair_initialization']['reference_r0_template_sha256']:
            raise ValueError('Scratch template or shared Flow differs from matched baseline')
        audit=init['matched_pair_initialization']; permitted=['flow.fullres_projection.weight','flow.fullres_projection.bias'] if a.arm=='s5' else []
        if set(audit['missing_variant_keys'])!=set(permitted) or audit['unexpected_keys'] or audit['shape_mismatch_keys']:raise ValueError('Shared-parameter initialization mismatch')
        atomic_json(a.output/'initialization_match.json',dict(status='PASS',shared_flow_sha256=init['shared_flow_initialization_sha256'],reference_template_sha256=audit['reference_r0_template_sha256'],added_keys=permitted))
        run('train2k',[train,'--config',cfg,'--output',formal,'--max-updates','1000'])
        ck=formal/'checkpoints/resume_latest.pt'
        run('dev2k',[dev,'--config',cfg,'--checkpoint',ck,'--output',a.output/'dev_2k','--gpu',gpu])
        run('train5k',[train,'--config',cfg,'--output',formal,'--max-updates','2500','--resume',ck])
        run('dev5k',[dev,'--config',cfg,'--checkpoint',ck,'--output',a.output/'dev_5k','--gpu',gpu])
        run('baseline_dev5k',[dev,'--config',CONTROL/'configs/research_v2/hier_occ_flow/gctv_matched_original_fm_20k.yaml','--checkpoint',CONTROL/'experiments/desescene_gctv_matched_control/run/formal/checkpoints/step002500.pt','--output',a.output/'baseline_dev_5k','--gpu',gpu])
        atomic_json(a.output/'status.json',dict(status='COMPLETE_5K_AWAITING_COHORT_SCREEN',pid=os.getpid(),arm=a.arm,gpu=gpu,source_sha256=pins))
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='BLOCKED',arm=a.arm,gpu=gpu,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),source_sha256=pins));raise

if __name__=='__main__':main()
