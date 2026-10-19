"""180K-source +1080K-GT-context Teacher: full network and CD forward/backward."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
import torch
import yaml
from export_stage2_snapshot import admit_gpu1
from fpsgen.ops.gpu_kdtree import load_extension
from fpsgen.ops.chamfer import Chamfer3DDist
from fpsgen.utils.gpu_kdtree_training import chamfer_sqrt_gpu_kdtree
from fpsgen.voxel_point.metrics import chamfer_sqrt
from fpsgen.voxel_point.mapping_teacher import MappingTeacher
from fpsgen.voxel_point.dataset import VoxelPointDataset
from fpsgen.voxel_point.voxel_spec import VoxelGridSpec
from fpsgen.voxel_point.count_allocator import allocate_counts
from fpsgen.voxel_point.anchor_sampler import sample_gaussian_anchors
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.raw_gt import RawMapSource


def timed(fn):
    torch.cuda.synchronize();tick=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,time.perf_counter()-tick


class LargeContextProfilingTeacher(MappingTeacher):
    """Benchmark-only shape adaptation; unchanged net, weights and feature logic."""
    def forward(self, source, voxel_ids, gt_context):
        if not self.training or source.shape != (180000,3) or gt_context.shape != (1080000,3):
            raise ValueError('Profile requires train mode,180000 source,1080000 context')
        target_ids,_=self.spec.voxelize_points(gt_context)
        points=torch.cat((source,gt_context))
        centers=self.spec.voxel_centers(torch.cat((voxel_ids,target_ids))).to(points)
        flags=torch.cat((torch.zeros(len(source),device=source.device,dtype=torch.bool),
                         torch.ones(len(gt_context),device=source.device,dtype=torch.bool)))
        displacement=self.net.forward_conditional(points.float(),centers.float(),points.new_zeros(1).float(),known_flags=flags)
        if displacement.shape!=points.shape or not torch.isfinite(displacement).all():
            raise ValueError('Invalid Teacher displacement')
        return source+displacement[:len(source)]


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve previous results')
    report=dict(status='RUNNING',admission=admit_gpu1(),historical_cd_backend='keops',
        historical_step_seconds=1.4417583656664752,frame='08/003900',isolated_cd={},full_teacher={},
        scope='SingleGPU FP16 Teacher training-mode benchmark, no optimizer updates/checkpoint writes; historical checkpoint read for realistic endpoints only',
        criteria=dict(loss_absolute=1e-6,endpoint_gradient_relative_l2=1e-4,endpoint_gradient_absolute=1e-6))
    atomic_json(a.output,report);torch.set_num_threads(4)
    root=Path('/data-12/M2024-HWZ/FPSGen')
    run=root/'experiments/voxel_point/mapping_teacher_gt_source_100k_gpu23_20261015'
    cfg=yaml.safe_load((run/'frozen_config.yaml').read_text())
    geometry=yaml.safe_load((run/cfg['geometry']).read_text())['voxel'];spec=VoxelGridSpec(**geometry)
    data_root=Path(cfg['data']['root'])
    frame=dict(sequence='08',frame=3900,input=str(data_root/'08/input_/003900.npy'),gt=str(data_root/'08/gt_possion/003900.ply'))
    sample=VoxelPointDataset([frame],spec,gt_roi_policy='filter')[0]
    raw=RawMapSource(data_root,fast_viewpoint=True)
    full_gt=raw.load('08',3900)
    valid=spec.valid_points(torch.from_numpy(full_gt)).numpy()
    full_gt=np.ascontiguousarray(full_gt[valid],dtype=np.float32)
    rng=np.random.default_rng(20261019)
    selection=np.sort(rng.choice(len(full_gt),1080000,replace=False))
    gt=torch.from_numpy(full_gt[selection]).cuda()
    report.update(context_protocol='Exactly1080000 original raw-map points chosen without replacement;180000 source anchors from corresponding Poisson GT density; profiling only, no training update',
        context_seed=20261019,original_valid_raw_context_points=len(full_gt),context_shape=list(gt.shape))
    atomic_json(a.output,report)
    generator=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    counts=allocate_counts(sample['gt_counts'].to(device='cuda',dtype=torch.float32),spec,generator=generator,total_points=180000)
    anchors,ids=sample_gaussian_anchors(spec,counts,generator=generator)
    model=LargeContextProfilingTeacher(spec,cfg['network']).cuda().train()
    ckpt=run/'checkpoint_050000.pt';saved=torch.load(ckpt,map_location='cpu')
    state=saved.get('model',saved.get('state_dict'))
    if state is None:raise ValueError('Unrecognized Teacher checkpoint')
    model.load_state_dict(state,strict=True);del saved
    extension=load_extension(root/'env/voxel_point_cache/gpu_kdtree_v1')
    native=Chamfer3DDist()
    def native_loss(p,g):
        with torch.autocast('cuda',enabled=False):
            dp,dg,_,_=native(p.float()[None],g.float()[None])
            return .5*(dp.sqrt().mean()+dg.sqrt().mean())
    functions=dict(keops=lambda p,g:chamfer_sqrt(p,g,backend='keops'),
        chamfer_cuda=native_loss,gpu_kdtree=lambda p,g:chamfer_sqrt_gpu_kdtree(p,g,extension=extension))
    report.update(parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=file_sha(ckpt),
        source_points=len(anchors),gt_context_points=len(gt),network_input_points=len(anchors)+len(gt))
    atomic_json(a.output,report)
    report.update(status='INITIAL_TEACHER_FORWARD');atomic_json(a.output,report)
    torch.manual_seed(2901);torch.cuda.manual_seed_all(2901)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        endpoint=model(anchors,ids,gt).float().detach()
    reference_gradient=None;reference_loss=None
    # Full-size warmup per backend, followed by3 timed CD-only graph builds/backward.
    for name,fn in functions.items():
        report.update(status='TIMING_'+name);atomic_json(a.output,report)
        values=[]
        for repeat in range(4):
            p=endpoint.clone().requires_grad_();g=gt.detach()
            with torch.autocast('cuda',enabled=False):loss,forward=timed(lambda:fn(p,g))
            gradient,backward=timed(lambda:torch.autograd.grad(loss,p)[0])
            if not torch.isfinite(gradient).all():raise ValueError('Nonfinite CD gradient')
            if repeat==3:
                if name=='keops':reference_gradient=gradient.detach().clone();reference_loss=float(loss)
                relative=float((gradient-reference_gradient).norm()/reference_gradient.norm().clamp_min(1e-30))
                absolute=float((gradient-reference_gradient).abs().max())
                error=abs(float(loss)-reference_loss)
                report.setdefault('gradient_parity',{})[name]=dict(loss_absolute_error=error,gradient_relative_l2=relative,gradient_max_absolute=absolute,
                    pass_test=error<=1e-6 and relative<=1e-4 and absolute<=1e-6)
            row=dict(repeat=repeat,warmup=repeat==0,loss=float(loss),cd_forward_seconds=forward,cd_backward_seconds=backward)
            values.append(row);print('Isolated',name,row,flush=True)
        report['isolated_cd'][name]=dict(runs=values,median_forward_seconds=statistics.median(r['cd_forward_seconds'] for r in values[1:]),
            median_backward_seconds=statistics.median(r['cd_backward_seconds'] for r in values[1:]))
        atomic_json(a.output,report)
    del reference_gradient,endpoint
    # Identical seeds/weights for every backend; true train mode+activation checkpointing.
    # Split chain rule: CD dL/dEndpoint then network backward with the same scaled gradient.
    for name,fn in functions.items():
        report.update(status='TIMING_'+name);atomic_json(a.output,report)
        values=[]
        for repeat in range(4):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(2901);torch.cuda.manual_seed_all(2901)
            report.update(status='TEACHER_FORWARD_'+name,repeat=repeat);atomic_json(a.output,report)
            with torch.autocast('cuda',dtype=torch.float16):
                endpoint,forward=timed(lambda:model(anchors,ids,gt))
            endpoint_fp=endpoint.float()
            loss,cd_forward=timed(lambda:fn(endpoint_fp,gt))
            gradient,cd_backward=timed(lambda:torch.autograd.grad(loss*128.,endpoint_fp)[0])
            report.update(status='TEACHER_BACKWARD_'+name,repeat=repeat);atomic_json(a.output,report)
            _,backward=timed(lambda:endpoint_fp.backward(gradient))
            finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            row=dict(repeat=repeat,warmup=repeat==0,loss=float(loss.detach()),teacher_forward_seconds=forward,
                cd_forward_seconds=cd_forward,cd_backward_seconds=cd_backward,teacher_backward_seconds=backward,
                total_seconds=forward+cd_forward+cd_backward+backward,parameter_gradients_finite=finite)
            values.append(row);print('FullTeacher',name,row,flush=True)
            del loss,gradient,endpoint,endpoint_fp
            if not finite:raise ValueError('Teacher parameter gradient nonfinite')
        report['full_teacher'][name]=dict(runs=values,medians={k:statistics.median(r[k] for r in values[1:]) for k in (
            'teacher_forward_seconds','cd_forward_seconds','cd_backward_seconds','teacher_backward_seconds','total_seconds')})
        atomic_json(a.output,report)
    report.update(status='PASS' if all(r['pass_test'] for r in report['gradient_parity'].values()) else 'PARITY_REVIEW_REQUIRED',
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        source_hashes={p.name:file_sha(p) for p in [HERE.parents[2]/'fpsgen/utils/gpu_kdtree_training.py',Path(__file__)]})
    atomic_json(a.output,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        destination=Path(sys.argv[sys.argv.index('--output')+1])
        value=json.loads(destination.read_text()) if destination.exists() else {}
        value.update(status='CUDA_OOM',failed_stage=value.get('status'),error=str(error),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            point_count_reduced=False,other_jobs_stopped=False)
        atomic_json(destination,value)
        print(json.dumps(value,indent=2),flush=True)
        raise

