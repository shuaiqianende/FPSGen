"""XYZ-only deployment entry point; input manifest has no GT fields."""

import json

import torch

from common import parser, read_config, configure_runtime, load_checkpoint, file_sha, save_json
from fpsgen.voxel_point.dataset import load_xyz
from fpsgen.voxel_point.voxdet_density import VoxDetDensity
from fpsgen.voxel_point.point_flow import PointFlow
from fpsgen.voxel_point.pipeline import VoxelPointPipeline, save_inference


def main():
    p = parser(__doc__)
    p.add_argument('--input-manifest',required=True,type=str,help='JSON list: id and input path ONLY')
    args = p.parse_args()
    cfg,spec,sha = read_config(args.config)
    device,_ = configure_runtime()
    torch.set_num_threads(4)
    frames = json.loads(open(args.input_manifest).read())
    if not frames or any(set(row) != {'id','input'} for row in frames):
        raise ValueError('Inference manifest must contain ONLY id and input; GT paths are forbidden')
    if len({row['id'] for row in frames}) != len(frames):
        raise ValueError('Duplicate inference frame IDs')
    if args.max_frames:
        frames = frames[:args.max_frames]
    density = VoxDetDensity(spec,**cfg['model']).to(device)
    student = PointFlow(spec,cfg['network']).to(device)
    stage1 = load_checkpoint(density,cfg['stage1_checkpoint'],spec,stage='stage1')
    stage2 = load_checkpoint(student,cfg['student_checkpoint'],spec,stage='student')
    if not args.smoke and (stage1['mechanics_smoke_only'] or stage2['mechanics_smoke_only']):
        raise ValueError('Smoke checkpoints cannot be used for formal inference')
    pipeline = VoxelPointPipeline(density,student,spec,steps=cfg['sampling']['euler_steps'])
    provenance = {'config_sha256':sha,'stage1_checkpoint_sha256':file_sha(cfg['stage1_checkpoint']),
        'student_checkpoint_sha256':file_sha(cfg['student_checkpoint']),
        'physical_visible':__import__('os').environ['CUDA_VISIBLE_DEVICES'],
        'mechanics_smoke_only':args.smoke,'euler_steps':cfg['sampling']['euler_steps']}
    args.output.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    summaries = []
    for index,row in enumerate(frames):
        from pathlib import Path
        frame_id = str(row['id'])
        if Path(frame_id).name != frame_id or frame_id in ('.','..'):
            raise ValueError('Frame id must be one safe directory name')
        if Path(row['input']).parent.name != 'input_' or Path(row['input']).suffix != '.npy':
            raise ValueError('Official inference protocol reads only paired input_/*.npy')
        xyz = load_xyz(row['input']).to(device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['amp']):
            result = pipeline(xyz,seed=cfg['seed']+index)
        save_inference(args.output/frame_id,result,seed=cfg['seed']+index,spec=spec,provenance={**provenance,'input':row['input']})
        summaries.append({'id':frame_id,'final_shape':list(result['final_points'].shape),
            'density_finite':bool(torch.isfinite(result['density']).all()),'allocated_points':int(result['counts'].sum()),
            'final_finite':bool(torch.isfinite(result['final_points']).all()),'timing_ms':result['timing_ms']})
        print(frame_id,result['timing_ms'],flush=True)
    save_json(args.output/'inference_summary.json',{'status':'PASS','gt_used':False,'teacher_used':False,
        'sinkhorn_used':False,'stage1_hidden_features_passed':False,'stage1_network_evaluations_per_frame':1,
        'peak_vram_bytes':torch.cuda.max_memory_allocated(device),'mechanics_smoke_only':args.smoke,
        'provenance':provenance,'frames':summaries})


if __name__ == '__main__':
    main()
