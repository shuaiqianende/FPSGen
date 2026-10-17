"""Admit before CUDA; measure actual384 network, never substitute static shapes."""
from train_stage1_occ import load_config
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy,ids_to_dense_occupancy,occupancy_flow_pair,masked_occ_loss
from fpsgen.hier_occ_flow.stage1_monitoring import gradient_stats
from fpsgen.hier_occ_flow.stage1_network import occupancy_lidar_features
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.dataset import input_xyz


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,choices=(2,3),required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();gate=gpu_preflight((args.physical_gpu,));gate['executed_gpu_operations']=False
    if gate['status']!='AVAILABLE' or not args.execute: atomic_json(args.output,gate);return
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
    cfg=load_config(args.config);g=validate_config(cfg);ds=SparseShardDataset(cfg['train_dataset'],g,verify=True)
    indices=sorted({int(np.argmin(abs(ds.index['length']-np.percentile(ds.index['length'],q)))) for q in (50,90,99,100)})
    torch.manual_seed(cfg['training']['seed']);model=Stage1DirectOccupancy(g,objective=cfg['model']['objective'],widths=cfg['model']['widths']).cuda().train()
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    scaler=torch.cuda.amp.GradScaler(init_scale=cfg['training'].get('amp_initial_scale',16384.))
    rows=[];batch=cfg['training']['batch_per_gpu']
    for i in indices:
        record=ds[i];xyz=torch.from_numpy(input_xyz(Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy")).cuda()
        cpu_features,cpu_indices=occupancy_lidar_features(xyz.cpu(),model.geometry)
        gpu_features,gpu_indices=occupancy_lidar_features(xyz,model.geometry)
        coordinate_pass=torch.equal(cpu_indices[0],gpu_indices[0].cpu())
        feature_error=float((cpu_features-gpu_features.cpu()).abs().max())
        target=ids_to_dense_occupancy(record['parent_ids'],g,device='cuda').expand(batch,-1,-1,-1,-1).contiguous()
        batch_xyz=xyz if batch==1 else [xyz for _ in range(batch)]
        pair=occupancy_flow_pair(target,model.valid,generator=torch.Generator(device='cuda').manual_seed(77)) if model.objective=='fm' else None
        shapes=[]
        hook=model.encoder.register_forward_hook(lambda module,args,output:shapes.extend([list(v.shape) for v in output]))
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):
            prediction=model(batch_xyz,state=pair['state'],time=pair['time']) if pair is not None else model(batch_xyz)
            loss=masked_occ_loss(prediction,pair['velocity'] if pair is not None else target,model.valid,objective=model.objective)
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.cuda.synchronize();hook.remove()
        grads={name:gradient_stats(module) for name,module in (('encoder',model.encoder),('flow',model.flow),('head',model.flow.output))}
        rows.append({'sequence':record['sequence'],'frame':record['frame'],'occupied_parents':len(record['parent_ids']),
            'condition_shapes':shapes,'output_shape':list(prediction.shape),'loss':float(loss),
            'gradients':grads,'amp_scale':scaler.get_scale(),'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cpu_gpu_coarse_indices_exact':coordinate_pass,'cpu_gpu_feature_max_abs_error':feature_error})
        scaler.update();model.zero_grad(set_to_none=True)
    expected=[[batch,128,16,192,192],[batch,128,8,96,96],[batch,128,4,48,48]]
    passed=all(r['condition_shapes']==expected and r['output_shape']==[batch,1,32,384,384]
        and r['cpu_gpu_coarse_indices_exact'] and r['cpu_gpu_feature_max_abs_error']<1e-5
        and np.isfinite(r['loss']) and all(v['l2'] and v['nonfinite_elements']==0 for v in r['gradients'].values()) for r in rows)
    gate.update(status='PASS' if passed else 'FAIL',executed_gpu_operations=True,geometry_sha256=g.sha256,rows=rows,batch_per_gpu=batch,
        maximum_scope='dataset maximum' if len(ds)==17870 else 'sample cohort maximum only')
    atomic_json(args.output,gate)

if __name__=='__main__':main()
