"""Same predicted Stage1 support, all valid Children occupied: native-grid control."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,hashlib
import numpy as np
import torch
from scripts.research_v2.joint_fm.evaluate import checkpoint,REFERENCE
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.meanflow_sampling import sample_meanflow,stage1_score
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.joint_fm_inference import predicted_support,ids_sha
from fpsgen.hier_occ_flow.voxelization import unpack_children,lookup_masks,occupied_fine_coords,pack_children
from fpsgen.hier_occ_flow.evaluation import rates
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--continuous',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('Admitted GPU3 only')
    gpu=gpu_preflight((3,),minimum_free_gib=22.,allow_shared_compute=False)
    if gpu['status']!='AVAILABLE':raise RuntimeError(json.dumps(gpu))
    if a.output.exists():raise FileExistsError('Preserve previous control')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg,saved,proof=checkpoint('S1-FM-Scratch20K');g=SparseHierarchyGeometry(**cfg['geometry'])
    model=Stage1DirectOccupancy(g,objective='fm',widths=cfg['model']['widths'],activation_checkpointing=False).cuda()
    model.load_state_dict({k[len('stage1.'):]:v for k,v in saved['ema'].items()},strict=True);del saved
    apply_explicit_policy(model);model.eval().requires_grad_(False);rows=[]
    if a.continuous:
        from fpsgen.hier_occ_flow.raw_gt import RawMapSource
        from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
        from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
        from unified_pointcloud_metrics import evaluate_prediction
        _,sha=raw_source_snapshot(cfg['data_root'],[('08',f) for f in range(0,4001,100)])
        if sha!=json.loads((REFERENCE/'protocol.json').read_text())['raw_source_manifest_sha256']:raise ValueError('Frozen rawGT source mismatch')
        raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=g)
    with torch.no_grad():
        for frame in range(0,4001,100):
            path=Path(cfg['data_root'])/'08/input_'/f'{frame:06d}.npy';meta=json.loads((REFERENCE/'targets'/f'08_{frame:06d}.json').read_text())
            if file_sha(path)!=meta['input_sha256']:raise ValueError('Original partial scan changed')
            xyz=torch.from_numpy(input_xyz(path)).cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                with torch.autocast('cuda',enabled=False):pyramid=model.encoder([xyz])
                obs=model.geometry.observed_mask(xyz);valid=model.valid
                source=torch.randn(valid.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',frame,20261031)))*valid
                fn=lambda x,t,s:model.flow(x,obs,pyramid,t)*valid
                endpoint=sample_meanflow(fn,source,valid,nfe=2,scenes=1);ids,_=predicted_support(stage1_score(endpoint)*valid,valid,g)
            # GT only measures the generated support; never affects generation.
            with np.load(REFERENCE/'targets'/f'08_{frame:06d}.npz',allow_pickle=False) as data:
                gt_ids=data['parent_ids'];masks=data['child_masks'];targets=unpack_children(lookup_masks(gt_ids,masks,ids.cpu().numpy()),g)
                total=int(unpack_children(masks,g).sum())
            pred_valid=g.valid_children(ids.cpu().numpy());tp=int((targets&pred_valid).sum());count=int(pred_valid.sum())
            row=dict(frame=frame,parent_ids_sha256=ids_sha(ids),parent_count=len(ids),fine=rates(dict(tp=tp,fp=count-tp,fn=total-tp)),
                     occupied_child_count=count,extra_parent_count=0)
            if a.continuous:
                gt=raw.load('08',frame);_,inside=g.voxelize_points(gt);target=np.ascontiguousarray(gt[inside],dtype=np.float32)
                fine=occupied_fine_coords(ids.cpu().numpy(),pack_children(pred_valid,g),g)
                points=g.centers(fine,fine=True).astype(np.float32)
                row['continuous']=evaluate_prediction(points,target,g,metric,device='cuda')
            rows.append(row);print(f'Support-fill control {len(rows)}/41',flush=True)
    summary={k:float(np.mean([r['fine'][k] for r in rows])) for k in ('precision','recall','f1','iou')}
    if a.continuous:summary.update({k:float(np.mean([r['continuous'][k] for r in rows])) for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')})
    atomic_json(a.output/'results.json',dict(status='EVALUATED_NATIVE_SUPPORT_FILL41',
        scope='Stage1 predicted Parents expanded to all valid eight Children; not a learned Fine prediction; no Stage2',
        checkpoint=proof,gpu=gpu,rows=rows,summary=summary))


if __name__=='__main__':main()
