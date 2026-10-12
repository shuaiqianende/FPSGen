"""Reproduce reviewed source extraction at explicitly pinned upstream commits."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PINS = {'VoxDet': 'e10d8280de96d4f03f54b616e57e7c6754044fa4', 'PWM': 'c6e809486db3d0b6f8503318dfea246c06c7a50e'}
VOX_CLASSES = {
    'models/backbones/lidar_encoder.py': {'LidarEncoder'},
    'models/backbones/resnet3d.py': {'BasicBlock3D', 'CustomResNet3D'},
    'models/necks/generalizedfpn.py': {'GeneralizedLSSFPN'},
    'models/necks/spatial_deocupled_fpn_dense.py': {'DensePooling', 'SpatiallyDecoupledFPN', 'TPVDecoupleModule'},
    'models/dense_heads/det_head.py': {'VoxelAttentionAggregation', 'VoxDetHead', 'create_grid', 'generate_grid'},
    'models/dense_heads/occ_head.py': {'OccHead'},
}
HEADER = '''# Generated from the pinned official VoxDet source; see SOURCE_MANIFEST.json.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch_scatter
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from timm.layers import DropPath
from torchvision.ops import DeformConv2d
'''


def main():
    manifest = {'upstream': PINS, 'files': [], 'adaptations': [
        'VoxDet: extract only configured LiDAR classes, remove MMDetection registry decorators and unused semantic loss methods; class init/forward unchanged except documented grid patch.',
        'VoxDet create_grid: XYZ tensor grid_sample uses Z,Y,X coordinate order; align_corners=False cell-center normalization.',
        'PWM: import namespace relocated only; official PTv3 trunk/time conditioning/adapter unchanged.',
    ]}
    vendor = ROOT / 'fpsgen/voxel_point/vendor'
    for name, pin in PINS.items():
        upstream = ROOT / 'env/src' / name
        head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if head != pin:
            raise ValueError(f'{name} commit mismatch: {head}')
    (vendor / '__init__.py').write_text('')
    (vendor / 'voxdet/__init__.py').write_text('')
    (vendor / 'pwm/__init__.py').write_text('')
    for relative, names in VOX_CLASSES.items():
        source = ROOT / 'env/src/VoxDet/mmdet3d_plugin' / relative
        original = source.read_text()
        tree = ast.parse(original)
        nodes = []
        for node in tree.body:
            if getattr(node, 'name', None) not in names:
                continue
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
                if node.name in {'VoxDetHead', 'OccHead'}:
                    node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name in {'__init__', 'forward'}]
            if getattr(node, 'name', None) == 'create_grid':
                node = ast.parse('''def create_grid(grid_x, grid_y, grid_z, X, Y, Z, device):
    # grid_sample input spatial dimensions are X,Y,Z; coordinates are W,H,D.
    return torch.stack((2 * (grid_z + .5) / Z - 1,
                        2 * (grid_y + .5) / Y - 1,
                        2 * (grid_x + .5) / X - 1), dim=-1)
''').body[0]
            nodes.append(node)
        if {n.name for n in nodes} != names:
            raise ValueError(f'Missing configured class in {source}')
        destination = vendor / 'voxdet' / Path(relative).name
        header = HEADER
        if relative.startswith('models/necks/'):
            header = header.replace('from mmcv.runner import BaseModule', 'from mmengine.model import BaseModule')
        destination.write_text(header + '\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[])) + '\n')
        raw = ROOT / 'third_party/voxel_point/VoxDet' / relative
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, raw)
        manifest['files'].append({'upstream_path': 'mmdet3d_plugin/' + relative,
            'upstream_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'adapted_path': str(destination.relative_to(ROOT)),
            'adapted_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    for name in ['ptv3_reference_adapter.py', 'ptv3_time_conditioning.py']:
        source = ROOT / 'env/src/PWM/pointworldmodel/core/point' / name
        destination = vendor / 'pwm' / name
        destination.write_text(source.read_text().replace('pointworldmodel.core.point.', 'fpsgen.voxel_point.vendor.pwm.'))
        manifest['files'].append({'upstream_path': 'pointworldmodel/core/point/' + name,
            'upstream_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'adapted_path': str(destination.relative_to(ROOT)),
            'adapted_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    reference = ROOT / 'env/src/PWM/pointworldmodel/core/point/ptv3_reference'
    shutil.copytree(reference, vendor / 'pwm/ptv3_reference', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for source in sorted(reference.rglob('*.py')):
        destination = vendor / 'pwm/ptv3_reference' / source.relative_to(reference)
        manifest['files'].append({'upstream_path': 'pointworldmodel/core/point/ptv3_reference/' + str(source.relative_to(reference)),
            'upstream_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'adapted_path': str(destination.relative_to(ROOT)),
            'adapted_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    for repo, files in {'VoxDet': ['LICENSE', 'configs/voxdet-semnatickitti-lidar.py', 'mmdet3d_plugin/models/detectors/VoxDetLiDAR.py', 'mmdet3d_plugin/models/tpvbranch/LocalAggregator.py', 'mmdet3d_plugin/models/tpvbranch/fuser.py', 'mmdet3d_plugin/datasets/pipelines/loading_kitti_lidar.py'], 'PWM': ['pointworldmodel/core/point/mapping_teacher.py','pointworldmodel/core/point/point_flow_model.py','pointworldmodel/core/point/anchor.py','pointworldmodel/core/voxel/grid.py','pointworldmodel/core/point/endpoint_refinement/sinkhorn.py','pointworldmodel/core/point/endpoint_refinement/sparse_global_ot.py','pointworldmodel/core/point/endpoint_refinement/knn.py']}.items():
        for name in files:
            source = ROOT / 'env/src' / repo / name
            destination = ROOT / 'third_party/voxel_point' / repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            manifest['files'].append({'upstream_path': repo + '/' + name,'upstream_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'reference_only': True})
    (ROOT / 'third_party/voxel_point/SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
