"""CPU construction audit with historical checkpoint reads forbidden."""
from train_stage1_occ import load_config
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import torch
from fpsgen.hier_occ_flow.stage1_training import validate_config,state_sha
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=load_config(args.config);g=validate_config(cfg);rows=[]
    for objective in ('fm','bce'):
        torch.manual_seed(cfg['training']['seed'])
        with patch('torch.load',side_effect=RuntimeError('External weights prohibited during construction')):
            model=Stage1DirectOccupancy(g,objective=objective,widths=cfg['model']['widths'])
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters())
        row={'objective':objective,'encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),
            'flow_parameters':sum(p.numel() for p in model.flow.parameters()),'all_online_parameters_trainable':all(p.requires_grad for p in model.parameters()),
            'model_sha256':state_sha(model),'ema_sha256':state_sha(ema),'optimizer_state_entries':len(optimizer.state),
            'pretrained_parameters_loaded':0,'tokenizer_grid':model.encoder.lidar_tokenizer.grid_size,
            'mask_shape':list(model.valid.shape),'valid_parent_cells':int(model.valid.sum()),
            'expected_condition_zyx':[[1,128,16,192,192],[1,128,8,96,96],[1,128,4,48,48]],
            'condition_shapes_measured':False,'full_network_forward_executed':False}
        assert row['model_sha256']==row['ema_sha256'] and row['optimizer_state_entries']==0
        rows.append(row);del model,ema,optimizer
    atomic_json(args.output,{'status':'PASS_CPU_CONSTRUCTION_ONLY','geometry_sha256':g.sha256,'rows':rows})

if __name__=='__main__': main()
