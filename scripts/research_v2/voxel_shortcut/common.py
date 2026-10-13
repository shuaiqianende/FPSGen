"""Independent configs, frozen split, local caches and permitted physical GPUs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from fpsgen.voxel_shortcut.geometry import ShortcutGeometry
from fpsgen.voxel_point.dataset import paired_frames,development_split


def parser(description,*,training=False):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-frames',type=int,default=0);p.add_argument('--smoke',action='store_true')
    if training:
        p.add_argument('--max-steps',type=int,default=0);p.add_argument('--pilot',action='store_true')
        p.add_argument('--initialize',type=Path);p.add_argument('--precision',choices=('fp32','fp16'))
        p.add_argument('--gate-report',type=Path)
    return p


def read_config(path):
    cfg=yaml.safe_load(path.read_text())
    geometry=yaml.safe_load((path.parent/cfg['geometry']).read_text())['voxel']
    spec=ShortcutGeometry(**geometry)
    if spec.shape!=(256,256,32) or spec.origin!=(-50.,-50.,-4.) or spec.maximum!=(50.,50.,4.4):
        raise ValueError('Production SS-VSF geometry must remain256x256x32 in the fixed50m space')
    if any(g not in (2,3) for g in cfg.get('runtime',{}).get('physical_gpu_ids',[2,3])):
        raise ValueError('Configuration may only name physical GPU2/3')
    if cfg.get('runtime',{}).get('batch_per_gpu',1)!=1:
        raise ValueError('Current launcher implements B1/GPU')
    cfg['voxel']=spec.to_dict()
    sha=hashlib.sha256(yaml.safe_dump(cfg,sort_keys=True).encode()).hexdigest()
    return cfg,spec,sha


def frames_for(cfg,split):
    if split=='test':
        if not cfg['evaluation'].get('protocol_frozen',False):
            raise ValueError("Freeze development protocol before seq08 access")
        return paired_frames(cfg['data']['root'],['08'])
    manifest=json.loads(Path(cfg['data']['split_manifest']).read_text())
    frames=manifest[split]
    if any(f['sequence']=='08' for f in frames):
        raise ValueError("seq08 train/development leakage")
    return frames


def select_spaced(frames,n):
    if n<0:raise ValueError('max-frames cannot be negative')
    if not n or n>=len(frames):
        return frames
    return [frames[int(i*(len(frames)-1)/max(n-1,1))] for i in range(n)]


def runtime():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in {'2','3','2,3'}:
        raise ValueError("Only physical GPU2/3 allowed")
    subprocess.run(['nvidia-smi','--id=2,3','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],check=True)
    cache=ROOT/'env/voxel_shortcut_cache'
    for key,name in [('TORCH_EXTENSIONS_DIR','torch'),('KEOPS_CACHE_FOLDER','keops'),('XDG_CACHE_HOME','xdg'),('PIP_CACHE_DIR','pip')]:
        os.environ[key]=str(cache/name)
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('LOCAL_RANK',0))
    os.environ['KEOPS_CACHE_FOLDER']=str(cache/'keops'/('visible_'+visible.replace(',','_'))/f'rank{rank}')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    world=int(os.environ.get('WORLD_SIZE',1))
    if world>1:
        torch.distributed.init_process_group('nccl')
    torch.set_num_threads(4)
    return torch.device('cuda',rank),rank,world


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load_base(cfg,spec,device,*,require_frozen=True):
    from fpsgen.voxel_shortcut.base_density import BaseDensity
    base=BaseDensity(spec,activation_checkpointing=cfg['runtime']['activation_checkpointing']).to(device)
    checkpoint=torch.load(cfg['base_checkpoint'],map_location='cpu')
    if checkpoint['stage']!='ssvsf_base' or checkpoint['geometry']!=spec.to_dict():
        raise ValueError("SS-VSF base checkpoint/protocol mismatch")
    base.load_state_dict(checkpoint['model'],strict=True)
    base.eval().requires_grad_(False)
    return base,sha_file(cfg['base_checkpoint']),checkpoint
