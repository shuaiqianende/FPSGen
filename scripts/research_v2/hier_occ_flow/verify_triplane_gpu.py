"""Full real maximum-parent Batch2 FM forward/backward and checkpoint audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,copy,json,os,time,traceback
import torch,yaml
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy,ids_to_dense_occupancy,occupancy_flow_pair,masked_occ_loss
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.trainer import atomic_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--v3-probe',action='store_true');a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());c['geometry']=yaml.safe_load((a.config.parent/c['geometry']).read_text());g=validate_config(c)
    if a.v3_probe:
        if c['model']['encoder_variant']!='triplane_v2':raise ValueError('V3 probe derives from V2 config only')
        c['model']['encoder_variant']='triplane_v3'
    gpu=c['training']['physical_gpus'][0]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Physical GPU mismatch')
    admission=gpu_preflight((gpu,),minimum_free_gib=22.,allow_shared_compute=False)
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if a.output.exists():raise FileExistsError('Prior audit preserved')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(c['training']['seed']);torch.cuda.manual_seed_all(c['training']['seed'])
    source_paths=[ROOT/'fpsgen/hier_occ_flow'/p for p in ('triplane_encoder.py','gctv_encoder.py','stage1_occupancy.py','stage1_training.py')]
    report=dict(status='RUNNING',config_sha256=file_sha(a.config),source_sha256={p.name:file_sha(p) for p in source_paths},gpu=admission,variant=c['model']['encoder_variant'],batch=2,gt_used_as_input=False)
    try:
        train=SparseShardDataset(c['train_dataset'],g,verify=False);dev=SparseShardDataset(c['development_dataset'],g,verify=False)
        selection=sorted([(int(row['length']),name,i) for name,ds in [('train',train),('development',dev)] for i,row in enumerate(ds.index)],reverse=True)[:2]
        records=[(train if name=='train' else dev)[i] for _,name,i in selection]
        points=[torch.from_numpy(input_xyz(Path(c['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).cuda() for r in records]
        model=Stage1DirectOccupancy(g,objective='fm',widths=c['model']['widths'],activation_checkpointing=True,encoder_variant=c['model']['encoder_variant']).cuda().train()
        optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-4},{'params':model.flow.parameters(),'lr':2e-4}],weight_decay=.01)
        scaler=torch.cuda.amp.GradScaler(init_scale=16384.);gen=torch.Generator(device='cuda').manual_seed(c['training']['seed']+1000)
        target=torch.cat([ids_to_dense_occupancy(r['parent_ids'],g,device='cuda') for r in records]);pair=occupancy_flow_pair(target,model.valid,generator=gen)
        event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):
            event0.record();pyramid=model.encoder(points);event1.record()
            pred=model(points,state=pair['state'],time=pair['time'],condition=pyramid)
            loss=masked_occ_loss(pred,pair['velocity'],model.valid,objective='fm')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norms={name:float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in module.parameters() if p.grad is not None]))) for name,module in [('encoder',model.encoder),('flow',model.flow)]}
        if not bool(torch.isfinite(loss)) or any(not 0<v<float('inf') for v in norms.values()):raise ValueError('Loss/gradient not finite and nonzero')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        initial=copy.deepcopy(model.state_dict());ema=copy.deepcopy(model).eval().requires_grad_(False)
        checkpoint=dict(model=initial,ema=ema.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),generator=gen.get_state())
        atomic_checkpoint(a.output/'checkpoint_roundtrip.pt',checkpoint);loaded=torch.load(a.output/'checkpoint_roundtrip.pt',map_location='cpu')
        if not all(torch.equal(v.cpu(),loaded['model'][k]) and torch.equal(v.cpu(),loaded['ema'][k]) for k,v in initial.items()):raise ValueError('EMA/checkpoint roundtrip mismatch')
        report.update(status='GPU_PASS',selected=selection,records=[dict(sequence=r['sequence'],frame=r['frame'],gt_parents=len(r['parent_ids'])) for r in records],
            points=[len(v) for v in points],pyramid_shapes=[list(v.shape) for v in pyramid],pyramid_dtypes=[str(v.dtype) for v in pyramid],loss=float(loss),gradient_norms=norms,
            encoder_event_ms=event0.elapsed_time(event1),step_wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_roundtrip=True)
    except Exception as e:report.update(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    atomic_json(a.output/'result.json',report);print(json.dumps(report,indent=2),flush=True)
    if report['status']!='GPU_PASS':raise SystemExit(2)

if __name__=='__main__':main()
