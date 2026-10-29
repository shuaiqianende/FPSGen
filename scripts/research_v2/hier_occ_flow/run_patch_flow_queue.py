"""Wait for existing per-card BC queue, then gate/smoke/train/evaluate."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight

BC=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-u1-bc-v1/experiments/desescene_u1_bc_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or a.output.exists():raise ValueError('wrong device or output exists')
    a.output.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    arm='psa' if a.gpu==2 else 'psab';dep=BC/('gpu2_queue' if a.gpu==2 else 'gpu3_shared_retry')/'status.json'
    cfg=ROOT/f'configs/research_v2/hier_occ_flow/patch_{arm}_20k.yaml';gate=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py'
    files=[cfg,gate,train,ev,Path(__file__),*[ROOT/'fpsgen/hier_occ_flow'/n for n in ('patch_fm_backbone.py','patch_unet_condition.py','stage1_occupancy.py','stage1_training.py','stage1_network.py','gctv_encoder.py')]];pins={str(x):file_sha(x) for x in files}
    def status(s,**kw):atomic_json(a.output/'status.json',dict(status=s,arm=arm,gpu=a.gpu,pid=os.getpid(),source_sha256=pins,**kw))
    def run(label,cmd):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned source changed')
        with (a.output/(label+'.log')).open('x') as f:
            c=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+label,child_pid=c.pid)
            if c.wait():raise RuntimeError(label+' failed; inspect log')
    try:
        status('WAITING_PRIOR_QUEUE',dependency=str(dep))
        while True:
            if dep.exists():
                s=json.loads(dep.read_text()).get('status','')
                if s=='COMPLETE':break
                if s=='BLOCKED':raise RuntimeError('prior queue blocked; inspect before continuing')
            time.sleep(20)
        status('WAITING_SHARED_GPU')
        while gpu_preflight((a.gpu,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
        run('gate',[gate,'--config',cfg,'--output',a.output/'gate'])
        smoke=a.output/'smoke';base=[train,'--config',cfg,'--output',smoke,'--smoke'];run('smoke2',base+['--max-updates','2']);run('smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
        s=json.loads((smoke/'summary.json').read_text());init=json.loads((smoke/'initialization_audit.json').read_text())
        if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34:raise ValueError('Smoke numerical health failed')
        from fpsgen.hier_occ_flow.patch_fm_backbone import PatchFlowUNet
        import torch
        with torch.random.fork_rng(devices=[]):torch.manual_seed(20261032+104729);model=PatchFlowUNet(activation_checkpointing=False)
        h=hashlib.sha256()
        for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.contiguous().numpy().tobytes())
        if init['shared_flow_initialization_sha256']!=h.hexdigest():raise ValueError('Paired Flow template mismatch')
        formal=a.output/'formal';run('train20k',[train,'--config',cfg,'--output',formal]);s=json.loads((formal/'summary.json').read_text())
        if s['successful_updates']!=10000 or s['successful_frame_exposures']!=20000:raise ValueError('20K incomplete')
        run('seq08_41',[ev,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',a.output/'seq08_41']);status('COMPLETE_20K_AND_41')
    except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
