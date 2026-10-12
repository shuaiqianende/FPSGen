"""GT is read in this separate evaluator, never in sample_pipeline."""

import json

import torch

from common import parser, read_config, configure_runtime, save_json
from fpsgen.voxel_point.dataset import load_xyz
from fpsgen.voxel_point.metrics import point_metrics


def main():
    p = parser(__doc__)
    p.add_argument('--evaluation-manifest',required=True,help='JSON list containing id, generated, gt')
    args = p.parse_args()
    cfg,spec,sha = read_config(args.config)
    device,_ = configure_runtime()
    torch.set_num_threads(4)
    rows = json.loads(open(args.evaluation_manifest).read())
    if args.max_frames:
        rows = rows[:args.max_frames]
    result = []
    for row in rows:
        generated = load_xyz(row['generated']).to(device)
        gt = load_xyz(row['gt']).to(device)
        if generated.shape != (180000,3):
            raise ValueError('Generated cloud must contain exactly 180000 XYZ')
        spec.counts(gt)
        result.append({'id':row['id'],**point_metrics(generated,gt,backend=cfg['runtime']['knn_backend'],spec=spec)})
    save_json(args.output/'point_evaluation.json',{'config_sha256':sha,'rows':result,'mechanics_smoke_only':args.smoke,
        'metric_contract':{'cd':'0.5 * (mean nearest euclidean pred→GT + GT→pred)','distance_units':'meters','fscore_thresholds_m':[.1,.2],'point_count':180000},
        'fd_u':'Run existing Utonia evaluator separately, frozen PCA-256 + sampling stability for small manifests'})


if __name__ == '__main__':
    main()
