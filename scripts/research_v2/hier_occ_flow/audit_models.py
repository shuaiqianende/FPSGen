"""CPU construction/strict-load audit, explicitly NOT a GPU forward result."""
from common import ROOT
import argparse
import importlib.metadata
from pathlib import Path
from unittest.mock import patch
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--stage1-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args(); torch.set_num_threads(4)
    rows=[]
    for n,split in ((256,(3,3,2)),(352,(2,2,2))):
        g=SparseHierarchyGeometry(parent_resolution=(n,n,32),child_split=split)
        for objective,backbone in (('bce','ptv3'),('fm','local'),('fm','ptv3')):
            torch.manual_seed(20261016)
            with patch('torch.load',side_effect=RuntimeError('External weights forbidden during Stage2 construction')):
                model=HierarchicalOccupancyModel(g,objective=objective,backbone=backbone)
            rows.append({'parent_resolution':g.parent_resolution,'K':g.k,'objective':objective,'backbone':backbone,
                'parameters':sum(p.numel() for p in model.parameters()),'all_trainable':all(p.requires_grad for p in model.parameters()),
                'external_weights_loaded':False,'cpu_constructor_passed':True,'gpu_forward_passed':False})
    from fpsgen.hier_occ_flow.condition import FrozenVoxDetCondition
    condition=FrozenVoxDetCondition(args.stage1_checkpoint)
    trunk=('lidar_tokenizer','voxel_backbone','voxel_neck','local_backbone','local_neck')
    packages={}
    for name in ('torch','mmcv-full','spconv-cu117','flash-attn','numpy'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]='not found under this distribution name'
    atomic_json(args.output,{'scope':'CPU construction and exact state_dict loading only','models':rows,
        'packages':packages,'stage1_checkpoint_sha256':file_sha(args.stage1_checkpoint),
        'stage1_load_strict':True,'stage1_total_density_parameters':sum(p.numel() for p in condition.parameters()),
        'stage1_trunk_including_both_local_neck_branches':sum(p.numel() for n in trunk for p in getattr(condition.encoder,n).parameters()),
        'stage1_all_frozen':all(not p.requires_grad for p in condition.parameters()),
        'PTv3':{'one_token_per_parent':True,'enc_channels':[64,128],'enc_depths':[2,2],'dec_depths':[2],
            'patch_sizes':[64,128],'pooling_stride':2,'grid_coord':'absolute parent XYZ indices, no scene-min origin',
            'serialization':['z','z-trans','hilbert','hilbert-trans'],'shuffle_orders':False,'point_norm':'instance',
            'flash_attention':True,'FM_time_injection':'all PTv3 blocks, 32D scene context','BCE_time_modulation':'not instantiated',
            'checkpointing':'tensor-only reentrant replay; no BatchNorm in checkpointed blocks'},
        'stage1_shape_forward_status':'BLOCKED_PENDING_GPU_ADMISSION','ptv3_permutation_gpu_status':'NOT_EXECUTED'})
    print(args.output)

if __name__=='__main__': main()
