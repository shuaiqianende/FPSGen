"""Compare existing metric artifacts on an identical frozen seq08 frame manifest."""

import json

from common import parser, read_config, save_json


def main():
    p = parser(__doc__)
    p.add_argument('--bevflow-report',required=True)
    p.add_argument('--voxel-report',required=True)
    args = p.parse_args()
    cfg,_,sha = read_config(args.config)
    if not cfg['evaluation']['threshold_frozen']:
        raise ValueError('Freeze configuration before seq08 comparison')
    reports = [json.loads(open(path).read()) for path in (args.bevflow_report,args.voxel_report)]
    keys = [set(row['id'] for row in value['rows']) for value in reports]
    if keys[0] != keys[1] or not keys[0]:
        raise ValueError('BEVFlow/VoxDet evaluation frame manifests differ')
    metrics = ('cd','fscore_0.1','fscore_0.2','coverage_0.1','coverage_0.2')
    comparison = {name:{metric:sum(row[metric] for row in report['rows'])/len(report['rows']) for metric in metrics}
                  for name,report in zip(('bevflow','voxdet_density_pointflow'),reports)}
    save_json(args.output/'comparison.json',{'config_sha256':sha,'frames':sorted(keys[0]),'metrics':comparison,
        'latency_note':'Use synchronized stage1/allocation/PointFlow/total measurements on the same physical GPU/environment.',
        'fd_u_note':'Attach existing frozen Utonia PCA-256 evaluation and sampling stability diagnostics.'})


if __name__ == '__main__':
    main()
