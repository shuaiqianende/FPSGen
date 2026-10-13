"""Parameter/provenance/GT-refine audit without changing upstream modules."""

import inspect
from pathlib import Path
import torch
from common import parser,read_config,save_json,sha_file
from fpsgen.voxel_shortcut.base_density import BaseDensity


def main():
    args=parser(__doc__).parse_args();cfg,spec,sha=read_config(args.config)
    model=BaseDensity(spec)
    initialize=model.initialize_density_checkpoint(cfg['initialize_density_checkpoint'])
    upstream=Path('env/src/VoxDet')
    ref=(upstream/'mmdet3d_plugin/models/detectors/VoxDetLiDAR.py').read_text()
    config=(upstream/'configs/voxdet-semnatickitti-lidar.py').read_text()
    save_json(args.output/'architecture_audit.json',{'official_commit':'e10d8280de96d4f03f54b616e57e7c6754044fa4',
        'config_sha256':sha,'parameters':sum(p.numel() for p in model.parameters()),'old_density_parameters':sum(p.numel() for p in model.encoder.parameters()),
        'added_occupancy_head_parameters':sum(p.numel() for p in model.occupancy_head.parameters()),
        'occupancy_design':'independent64-to-1 occupancy head on shared official refined features; positive count retains original1channel classifier',
        'preserved':['LidarEncoder6->128','CustomResNet3D[2,2,2]128','GeneralizedLSSFPN128','localResNet[3,3,3]128',
                     'SpatiallyDecoupledFPN','six-direction prediction','fourVoxelAttentionAggregation'],
        'official_config_use_gt_refine': 'use_gt_refine= True' in config,
        'official_gt_refine_role':'training target masking/car semantic run filters in detector; not used by new encoder/base forward',
        'new_inference_gt_arguments':False,'semantic_feature_used':False,
        'condition_layout':'threeZYX grids16x128x128,8x64x64,4x32x32;128channels',
        'legacy_voxdet_abi':'official trunk/head keepXYZinternally; one geometry adapter converts all public fields/conditions toZYX; new velocityConv3d usesZYX',
        'condition_role':'withinStage1 only; noStage2 export',
        'checkpoint_compatibility':'new encoder isolates BN buffers during nonreentrant recomputation; one running-stat update per forward; upstream module files unchanged',
        'density_warmstart':initialize,'upstream_detector_sha256':sha_file(upstream/'mmdet3d_plugin/models/detectors/VoxDetLiDAR.py')})


if __name__=='__main__':main()
