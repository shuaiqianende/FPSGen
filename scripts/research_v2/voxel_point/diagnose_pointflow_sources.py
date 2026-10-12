"""Training/development source analysis; GT-density oracle is never deployment."""

import torch

from common import parser, read_config, configure_runtime, frames_for, load_checkpoint, save_json
from train_core import sample_source
from fpsgen.voxel_point.dataset import VoxelPointDataset
from fpsgen.voxel_point.point_flow import PointFlow, euler_rollout
from fpsgen.voxel_point.metrics import point_metrics


def main():
    p = parser(__doc__)
    p.add_argument('--source-kind',choices=('gt_density','predicted_density'),default='gt_density')
    p.add_argument('--split',choices=('train','development'),default='development')
    args = p.parse_args()
    cfg,spec,sha = read_config(args.config)
    device,_ = configure_runtime()
    torch.set_num_threads(4)
    dataset = VoxelPointDataset(frames_for(cfg,args.split,args.max_frames),spec)
    student = PointFlow(spec,cfg['network']).to(device).eval()
    checkpoint = load_checkpoint(student,cfg['student_checkpoint'],spec,stage='student')
    if checkpoint['mechanics_smoke_only'] and not args.smoke:
        raise ValueError('Smoke model cannot enter formal source-quality analysis')
    generator = torch.Generator(device=device).manual_seed(cfg['seed'])
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            anchors,ids = sample_source(sample,cfg,spec,generator,device,
                use_predicted=args.source_kind=='predicted_density',formal=not args.smoke)
            torch.manual_seed(cfg['seed']+index)
            endpoint = euler_rollout(student,anchors,ids,steps=cfg['sampling']['euler_steps'])
            gt = sample['gt_xyz'].to(device)
            rows.append({'frame':sample['frame'],
                'anchors':point_metrics(anchors,gt,backend=cfg['runtime']['knn_backend'],spec=spec),
                'pointflow':point_metrics(endpoint,gt,backend=cfg['runtime']['knn_backend'],spec=spec)})
    save_json(args.output/'source_diagnostic.json',{'config_sha256':sha,'split':args.split,
        'source_kind':args.source_kind,'oracle_not_deployable':args.source_kind=='gt_density',
        'mechanics_smoke_only':args.smoke,'rows':rows})


if __name__ == '__main__':
    main()
