"""Frozen depth-moment U1+A Stage1 for Stage2; preserves legacy adapter."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from .stage1_occ_adapter import FrozenStage1OccupancyAdapter
from .geometry import SparseHierarchyGeometry
from .storage import file_sha
from .stage1_occupancy import Stage1DirectOccupancy

class FrozenT1Stage1Adapter(FrozenStage1OccupancyAdapter):
    def __init__(self,checkpoint,config,*,checkpoint_sha256,weights='ema'):
        nn.Module.__init__(self)
        checkpoint=Path(checkpoint).resolve(); config=Path(config).resolve()
        if file_sha(checkpoint)!=checkpoint_sha256:raise ValueError('T1 checkpoint SHA mismatch')
        cfg=json.loads(config.read_text());g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
        if SparseHierarchyGeometry(**cfg['geometry'])!=g or cfg['model'].get('encoder_variant')!='patch_depth' or cfg['model'].get('objective')!='fm':raise ValueError('T1 architecture or geometry mismatch')
        if weights!='ema':raise ValueError('T1 study uses final EMA')
        value=torch.load(checkpoint,map_location='cpu');ident=value.get('identity',{})
        if ident.get('config_sha256')!=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest():raise ValueError('T1 config/checkpoint identity mismatch')
        if ident.get('geometry_sha256')!=g.sha256 or value.get('initialization_mode')!='scratch' or value.get('parameter_sync_pass') is not True:raise ValueError('T1 lineage mismatch')
        if value.get('successful_updates')!=50000 or value.get('sampler',{}).get('successful_frame_exposures')!=100000:raise ValueError('T1 final 100K EMA required')
        with torch.random.fork_rng(devices=[]):self.model=Stage1DirectOccupancy(g,objective='fm',widths=cfg['model']['widths'],activation_checkpointing=False,encoder_variant='patch_depth')
        self.model.load_state_dict(value['ema'],strict=True);self.model.requires_grad_(False);self.eval();self.geometry=g
        self.encoder_forward_count=0;self.last_feature_shapes=None
        self.provenance={'type':'FrozenT1Stage1Adapter','checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha256,'config_sha256':ident['config_sha256'],'geometry_sha256':g.sha256,'weights':'ema','successful_updates':50000,'successful_frame_exposures':100000,'stage1_final_100k_frames':True,'score_layout':self.score_layout,'feature_layout':self.feature_layout,'input_feature_version':cfg['model']['input_feature_version']}
