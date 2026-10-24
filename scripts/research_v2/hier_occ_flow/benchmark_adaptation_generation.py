"""Paired preloaded-XYZ generation wall timing, separate from quality/profiling."""
from common import ROOT
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.encoder_adaptation import load_sources,load_stage2_source
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config,config_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.condition import sample_pyramid,observed_children
from fpsgen.hier_occ_flow.inference import sample
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import source_sha


def stats(values):
    x=np.asarray(values,dtype=np.float64)
    return {'mean':float(x.mean()),'std':float(x.std()),'median':float(np.median(x)),
        'p90':float(np.percentile(x,90)),'p95':float(np.percentile(x,95)),'min':float(x.min()),'max':float(x.max())}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reference',action='store_true');p.add_argument('--legacy-stage2',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in ('2','3'):raise ValueError('GPU2/3')
    admission=gpu_preflight((int(physical),))
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    torch.set_num_threads(4)
    if a.legacy_stage2:
        from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
        cfg,g=load_stage2_config(a.config);adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda()
        s1=adapter.model;stage2=Stage2OccupancyModel(g,**cfg['model']).cuda().eval()
        saved=torch.load(a.checkpoint,map_location='cpu')
        from fpsgen.hier_occ_flow.stage2_runtime import config_sha as legacy_config_sha
        if saved['identity']['config_sha256']!=legacy_config_sha(cfg) or saved['sampler']['successful_frame_exposures']!=20000:
            raise ValueError('Legacy timing requires its matching completed20K checkpoint')
        stage2.load_state_dict(saved['ema'],strict=True)
        policy=cfg
    else:
        cfg,g=load_adaptation_config(a.config);system,_,_=load_sources(cfg,torch.device('cuda'))
        if not a.reference:
            saved=torch.load(a.checkpoint,map_location='cpu')
            if (saved.get('initialization_mode')!='pretrained_ema_adaptation'
                or saved['identity']['config_sha256']!=config_sha(cfg) or saved['identity']['arm']!=cfg['arm']
                or saved['successful_frame_exposures']!=20000):raise ValueError('Matching completed20K adaptation required')
            system.load_state_dict(saved['ema'],strict=True)
        system.eval();s1=system.stage1;stage2=system.stage2
        if stage2 is None:stage2,_=load_stage2_source(cfg,g);stage2=stage2.cuda().eval()
        policy=json.loads(Path(cfg['reference_stage2_config']).read_text())
    s1.eval();stage2.eval()
    ds=SparseShardDataset(cfg['development_dataset'],g,verify=False)
    indices=np.linspace(0,len(ds)-1,25,dtype=int)
    frames=[]
    for i in indices:
        r=ds[int(i)];path=Path(cfg['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy"
        frames.append((int(i),r,torch.from_numpy(input_xyz(path)).cuda(),file_sha(path)))
    events={name:[] for name in ('encoder','flow','stage2')};handles=[]
    for name,module in (('encoder',s1.encoder),('flow',s1.flow),('stage2',stage2)):
        def before(m,args,name=name):
            event=torch.cuda.Event(enable_timing=True);event.record();events[name].append([event,None])
        def after(m,args,out,name=name):
            event=torch.cuda.Event(enable_timing=True);event.record();events[name][-1][1]=event
        handles.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after)))
    def generation(index,r,xyz):
        pyramid=s1.encode_condition(xyz)
        gen=torch.Generator(device='cuda').manual_seed(scene_seed(r['sequence'],r['frame'],policy['candidates']['seed']))
        score,_=s1.sample(xyz,nfe=2,generator=gen,condition=pyramid)
        ids_np,scores,_=select_occupancy_candidates(score[0,0].cpu().numpy(),g,**policy['candidates']['selection'])
        ids=torch.as_tensor(ids_np,device='cuda');centers=g.centers(g.unflatten(ids)).float()
        c={'parent_score':torch.as_tensor(scores,device='cuda'),
           'observed_child':torch.as_tensor(observed_children(xyz,ids_np,g),device='cuda'),
           'valid_child_mask':torch.as_tensor(g.valid_children(ids_np),device='cuda'),
           'lidar_features':sample_pyramid(pyramid,centers,g.origin,g.maximum).half()}
        gen=torch.Generator(device='cuda').manual_seed(20261031+index)
        _,occupied,_=sample(stage2,ids,c,nfe=4,generator=gen)
        selected=torch.nonzero(occupied);parents=g.unflatten(ids[selected[:,0]])
        return parents*2+torch.as_tensor(g.child_offsets(),device='cuda')[selected[:,1]]
    rows=[];torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for index,r,xyz,digest in frames[:5]:out=generation(index,r,xyz)
        for index,r,xyz,digest in frames[5:]:
            events={k:[] for k in events};torch.cuda.synchronize();tick=time.perf_counter()
            out=generation(index,r,xyz);torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
            if {k:len(v) for k,v in events.items()}!={'encoder':1,'flow':2,'stage2':4}:
                raise ValueError('Measured network call count differs from frozen NFE protocol')
            rows.append({'sequence':r['sequence'],'frame':r['frame'],'input_sha256':digest,
                'generation_wall_ms':wall,'fine_count':len(out),
                **{k+'_cuda_ms':sum(a.elapsed_time(b) for a,b in pairs) for k,pairs in events.items()}})
    for h in handles:h.remove()
    atomic_json(a.output,{'status':'COMPLETE','rows':rows,'summary':{k:stats([r[k] for r in rows])
        for k in ('generation_wall_ms','encoder_cuda_ms','flow_cuda_ms','stage2_cuda_ms')},
        'physical_gpu':int(physical),'gpu_admission':admission,'torch':torch.__version__,
        'config_sha256':config_sha(cfg),'source_sha256':source_sha(),
        'backend':'PyTorch eager originalAMP, PTv3 FP32 trunk / FP16 FlashAttention; same for every arm',
        'scope':'preloaded GPU XYZ to GPU native Fine coordinates; includes original CPU selector/observed operations; no disk IO/metrics/jitter',
        'warmup_frames':5,'timed_frames':20,'stage1_nfe':2,'stage2_nfe':4,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'checkpoint_sha256':file_sha(a.checkpoint) if a.checkpoint else None})


if __name__=='__main__':main()
