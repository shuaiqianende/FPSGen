"""Real 180k diagnostics; parameter selection is restricted to training scenes."""

import itertools

import torch

from common import parser, read_config, configure_runtime, frames_for, load_checkpoint, save_json
from fpsgen.voxel_point.dataset import VoxelPointDataset
from fpsgen.voxel_point.mapping_teacher import MappingTeacher
from fpsgen.voxel_point.metrics import point_metrics
from fpsgen.voxel_point.sinkhorn_refiner import refine_with_policy
from fpsgen.ops.endpoint_refinement.sparse_global_ot import knn_barycentre
from train_core import sample_source


def main():
    p = parser(__doc__)
    p.add_argument('--grid-search',action='store_true',help='Explicitly run registered finite training-only grid')
    p.add_argument('--predicted-sources',action='store_true')
    args = p.parse_args()
    cfg,spec,sha = read_config(args.config)
    device,_ = configure_runtime()
    torch.set_num_threads(4)
    frames = frames_for(cfg,'train',args.max_frames or (1 if args.smoke else 20))
    teacher = MappingTeacher(spec,cfg['network']).to(device).eval()
    checkpoint = load_checkpoint(teacher,cfg['teacher_checkpoint'],spec,stage='teacher')
    if checkpoint['mechanics_smoke_only'] and not args.smoke:
        raise ValueError('Untrained smoke Teacher cannot pass formal OT quality gate')
    generator = torch.Generator(device=device).manual_seed(cfg['seed'])
    settings = [cfg['sinkhorn']]
    if args.grid_search:
        d = cfg['diagnostic']
        settings = [{**cfg['sinkhorn'],'k':k,'epsilon':e,'iterations':it,'alpha':a}
                    for k,e,it,a in itertools.product(d['k'],d['epsilon'],d['iterations'],d['alpha'])]
    dataset = VoxelPointDataset(frames,spec)
    rows = []
    args.output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            gt = sample['gt_xyz'].to(device)
            anchors,ids = sample_source(sample,cfg,spec,generator,device,use_predicted=args.predicted_sources)
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['amp']):
                endpoint = teacher(anchors,ids,gt).float()
            nearest = knn_barycentre(endpoint,gt,k=1,backend=cfg['runtime']['knn_backend'])
            bary = knn_barycentre(endpoint,gt,k=16,backend=cfg['runtime']['knn_backend'])
            baselines = {name:point_metrics(p,gt,backend=cfg['runtime']['knn_backend'],spec=spec)
                         for name,p in [('source',anchors),('teacher',endpoint),('nearest_gt',nearest),('knn_barycenter',bary)]}
            for parameters in settings:
                refined,diagnostics = refine_with_policy(endpoint,gt,spec,backend=cfg['runtime']['knn_backend'],**parameters)
                row = {'frame':sample['frame'],'source_kind':'predicted_density' if args.predicted_sources else 'gt_density',
                       'parameters':parameters,'baselines':baselines,'diagnostics':diagnostics}
                rows.append(row)
                save_json(args.output/'sinkhorn_diagnostics.json',{'config_sha256':sha,
                    'mechanics_smoke_only':args.smoke,'pseudo_targets_reliable':False,
                    'gate_note':'Requires frozen 20 → 100 scene quality review on BOTH source kinds; never automatic from one smoke.', 'rows':rows})
                print(f"scene {index+1}/{len(dataset)} K{parameters['k']} balanced={diagnostics['balanced_ot_converged']} geometry={diagnostics['geometry_refinement_accepted']}",flush=True)


if __name__ == '__main__':
    main()
