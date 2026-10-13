"""GT-free SS-VSF inference; export only density/counts/anchor/voxel geometry."""

import json
from pathlib import Path
import numpy as np
import torch
from common import parser,read_config,runtime,save_json,load_base,sha_file
from fpsgen.voxel_point.dataset import load_xyz
from fpsgen.voxel_shortcut.dataset import validate_inference_frames
from fpsgen.voxel_shortcut.count_mapping import CountMapping
from fpsgen.voxel_shortcut.source_builder import build_source
from fpsgen.voxel_shortcut.shortcut_unet import ShortcutUNet
from fpsgen.voxel_shortcut.shortcut_flow import rollout
from fpsgen.voxel_shortcut.evaluator import endpoint_density
from fpsgen.voxel_shortcut.count_allocator import allocate_anchors


def main():
    p=parser(__doc__);p.add_argument('--input-manifest',required=True,type=Path);p.add_argument('--nfe',type=int,default=1)
    args=p.parse_args();cfg,spec,sha=read_config(args.config)
    rows=validate_inference_frames(json.loads(args.input_manifest.read_text()))
    device,rank,world=runtime()
    if world!=1:raise ValueError('Export one process per manifest')
    if not cfg['source_calibration']['frozen']:raise ValueError('Freeze development source gate')
    base,base_sha,_=load_base(cfg,spec,device);model=ShortcutUNet(**cfg['network']).to(device).eval()
    checkpoint=torch.load(cfg['flow_checkpoint'],map_location='cpu')
    if checkpoint['base_checkpoint_sha256']!=base_sha or checkpoint['geometry']!=spec.to_dict() or checkpoint['mapping']!=cfg['mapping'] or checkpoint['source']!=cfg['source']:
        raise ValueError('Frozen base/source/mapping mismatch')
    model.load_state_dict(checkpoint['ema'] if checkpoint['ema'] is not None else checkpoint['model'],strict=True)
    mapping=CountMapping(**cfg['mapping']);valid=spec.field_mask(device=device)
    with torch.inference_mode():
        for index,row in enumerate(rows[:args.max_frames] if args.max_frames else rows):
            if Path(row['id']).name!=row['id'] or row['id'] in ('.','..'):raise ValueError('Unsafe frame ID')
            xyz=load_xyz(row['input']).to(device);seed=cfg['seed']+index
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['precision']=='fp16'):
                frozen=base(xyz)
                source,_=build_source(frozen['density'],frozen['occupancy_probability'],spec.observed_mask(xyz),valid,mapping,
                    generator=torch.Generator(device=device).manual_seed(seed),**cfg['source'])
                if checkpoint['stage']=='residual':
                    if args.nfe!=1:raise ValueError('Residual baseline has one correction')
                    endpoint=source+model(source,0.,1.,frozen['condition'])*valid;detail={'actual_nfe':1}
                else:endpoint,detail=rollout(model,source,frozen['condition'],nfe=args.nfe,shortcut=checkpoint['stage'] in ('shortcut','surface'),valid=valid)
            density,raw,decode=endpoint_density(endpoint,mapping,valid)
            bridge=allocate_anchors(density.cpu(),spec,generator=torch.Generator().manual_seed(seed))
            folder=args.output/row['id'];folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'voxel_density.npz',density=density[0,0].cpu().numpy())
            np.savez_compressed(folder/'allocated_counts.npz',counts=bridge['counts'].numpy())
            for name in ('anchors','voxel_ids','voxel_centers'):
                filename='source_anchors' if name=='anchors' else name
                np.save(folder/f'{filename}.npy',bridge[name].numpy())
            save_json(folder/'manifest.json',{'id':row['id'],'input':row['input'],'seed':seed,'geometry':spec.to_dict(),
                'config_sha256':sha,'base_sha256':base_sha,'flow_sha256':sha_file(cfg['flow_checkpoint']),
                'density_layout':'ZYX','source_points':180000,'base_encodings':1,'actual_velocity_nfe':detail['actual_nfe'],
                'gt_used':False,'teacher_used':False,'sinkhorn_used':False,'stage2_hidden_features_exported':False,
                'pointflow_used':False,'pilot_only':checkpoint['pilot_only'],**decode})
            print(row['id'],'180000 anchors',detail['actual_nfe'],'velocityNFE',flush=True)


if __name__=='__main__':main()
