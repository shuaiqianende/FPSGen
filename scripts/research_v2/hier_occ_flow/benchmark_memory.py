"""Resource gate before allocating CUDA; full-scene benchmarks only when admitted."""
from common import ROOT
import argparse
import os
from pathlib import Path
import time
import yaml
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--physical-gpu',type=int,choices=(2,3),default=2); p.add_argument('--execute',action='store_true')
    p.add_argument('--config',type=Path); args=p.parse_args()
    gate=gpu_preflight((args.physical_gpu,)); gate['executed_gpu_operations']=False
    gate['static_stage1_shapes']={str(n):{'xyz_pyramid':[[n//(2**i),n//(2**i),32//(2**i)] for i in range(4)],
        'divisible_by_8':n%8==0,'full_forward_validated':False} for n in (256,320,352,384)}
    if gate['status']!='AVAILABLE' or not args.execute:
        atomic_json(args.output,gate); print(gate,flush=True); return
    if not args.config: raise ValueError('--execute requires config and actual prepared data')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
    # No Torch CUDA access occurs before admission above.
    import torch
    from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
    from fpsgen.hier_occ_flow.storage import SparseShardDataset
    from fpsgen.hier_occ_flow.dataset import assemble_scene
    from fpsgen.hier_occ_flow.condition import FrozenVoxDetCondition
    from fpsgen.hier_occ_flow.trainer import build_model
    from fpsgen.hier_occ_flow.flow_matching import training_pair,fm_loss,bce_loss
    cfg=yaml.safe_load(args.config.read_text()); cfg['geometry']=yaml.safe_load((args.config.parent/cfg['geometry']).read_text())
    g=SparseHierarchyGeometry(**cfg['geometry']); ds=SparseShardDataset(cfg['train_dataset'],g,verify=True)
    lengths=ds.index['length']; indices=sorted(set(int(__import__('numpy').argmin(abs(lengths-__import__('numpy').percentile(lengths,q)))) for q in (50,90,99,100)))
    encoder=FrozenVoxDetCondition(cfg['stage1_checkpoint']).cuda().eval()
    model=build_model(cfg,g).cuda().train(); rows=[]
    for i in indices:
        torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
        ids,c,target,source=assemble_scene(ds[i],cfg,g,torch.device('cuda'),encoder,20261016+i)
        pair=training_pair(target,c['valid_child_mask'],torch.Generator(device='cuda').manual_seed(41))
        with torch.autocast('cuda',dtype=torch.float16):
            if model.objective=='fm': loss=fm_loss(model(ids,state=pair['state'],time=pair['time'],**c),pair['velocity'],c['valid_child_mask'])
            else: loss=bce_loss(model(ids,**c),target,c['valid_child_mask'])
        loss.backward(); torch.cuda.synchronize()
        row={'scene':i,'tokens':len(ids),'loss':float(loss),'gradients_finite':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
            'gradient_nonzero':any(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters()),
            'forward_backward_seconds':time.perf_counter()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            kwargs={'state':pair['state'],'time':pair['time']} if model.objective=='fm' else {}
            original=model(ids,**kwargs,**c)
            perm=torch.randperm(len(ids),device='cuda'); shuffled={k:v[perm] for k,v in c.items()}
            if model.objective=='fm': kwargs={'state':pair['state'][perm],'time':pair['time']}
            reordered=model(ids[perm],**kwargs,**shuffled)[torch.argsort(perm)]
            error=float((original-reordered).abs().max())
        row.update(parent_permutation_max_abs_error=error,parent_permutation_pass=error<1e-4)
        rows.append(row); model.train()
    gate.update(status='PASS_GPU_SMOKE' if all(r['gradients_finite'] and r['gradient_nonzero'] and r['parent_permutation_pass'] for r in rows) else 'FAIL_GPU_SMOKE',
        executed_gpu_operations=True,rows=rows,scope='sample cohort token quantiles, not dataset-wide P99 guarantee')
    atomic_json(args.output,gate)

if __name__=='__main__': main()
