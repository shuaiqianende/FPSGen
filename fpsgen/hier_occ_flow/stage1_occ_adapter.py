"""Strict frozen384 Occupancy Stage1; no historical density loader is reachable."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .geometry import SparseHierarchyGeometry
from .storage import DATA_VERSION,file_sha


def validate_stage1_identity(value,config,geometry):
    identity=value.get('identity',{})
    expected={'experiment':'hs-occflow-stage1-occ384-v1','objective':'fm','geometry_sha256':geometry.sha256}
    if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('Not this384 binary Occupancy checkpoint')
    if value.get('initialization_mode')!='scratch' or value.get('parameter_sync_pass') is not True:
        raise ValueError('Stage1 must have verified scratch lineage and synchronized ranks')
    if config['geometry']!=geometry.to_dict():
        # JSON stores lists; geometry dataclass uses tuples.
        if SparseHierarchyGeometry(**config['geometry'])!=geometry:raise ValueError('Stage1 geometry mismatch')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if identity.get('config_sha256')!=digest:raise ValueError('Stage1 frozen config/checkpoint mismatch')
    if config['model'].get('objective')!='fm' or config['flow']!={'source':'gaussian','sigma':1.,'target':'binary_symmetric'}:
        raise ValueError('Stage1 must generate binary occupancy, not count density')
    if config['validation'].get('target_definition')!='raw_pre_poisson_surface_hit':
        raise ValueError('Stage1 target is not raw-map surface-hit occupancy')
    for split,key in (('train','train_dataset'),('development','development_dataset')):
        path=Path(config[key])/'manifest.json'
        if not path.is_absolute():path=Path(identity['output']).parents[2]/path
        # Caller resolves legacy relative dataset paths against the source worktree.
        if not path.exists():raise FileNotFoundError(f'Stage1 dataset manifest unavailable: {path}')
        manifest=json.loads(path.read_text())
        if file_sha(path)!=identity[f'{split}_manifest_sha256'] or manifest.get('data_version')!=DATA_VERSION or manifest['geometry_sha256']!=geometry.sha256:
            raise ValueError('Stage1 training data version/content mismatch')
    return identity


class FrozenStage1OccupancyAdapter(nn.Module):
    feature_layout='B,C,Z,Y,X'
    score_layout='Z,Y,X'
    feature_shapes=((1,128,16,192,192),(1,128,8,96,96),(1,128,4,48,48))

    def __init__(self,checkpoint,config,*,checkpoint_sha256,weights='ema'):
        super().__init__()
        checkpoint=Path(checkpoint).resolve();source=Path(config).resolve()
        if file_sha(checkpoint)!=checkpoint_sha256:raise ValueError('Stage1 checkpoint SHA changed')
        cfg=json.loads(source.read_text());original_cfg=json.loads(source.read_text())
        value=torch.load(checkpoint,map_location='cpu')
        geometry=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
        # Validation hashes the ORIGINAL config. Resolve dataset paths only while
        # opening them, using the source output's registered worktree location.
        validate_stage1_identity(value,original_cfg,geometry)
        if weights not in ('online','ema'):raise ValueError('Choose online or EMA explicitly')
        from .stage1_occupancy import Stage1DirectOccupancy
        with torch.random.fork_rng(devices=[]):
            self.model=Stage1DirectOccupancy(geometry,objective='fm',widths=cfg['model']['widths'],activation_checkpointing=False)
        state=value['ema'] if weights=='ema' else value['model']
        if state is None:raise ValueError('Requested Stage1 weights absent')
        self.model.load_state_dict(state,strict=True)
        self.model.requires_grad_(False);self.eval();self.geometry=geometry
        self.encoder_forward_count=0;self.last_feature_shapes=None
        self.provenance={'type':'FrozenStage1OccupancyAdapter','checkpoint':str(checkpoint),
            'checkpoint_sha256':checkpoint_sha256,'config_sha256':value['identity']['config_sha256'],
            'geometry_sha256':geometry.sha256,'data_version':DATA_VERSION,'weights':weights,
            'successful_updates':value['successful_updates'],
            'successful_frame_exposures':value['sampler']['successful_frame_exposures'],
            'stage1_final_100k_frames':value['sampler']['successful_frame_exposures']==100000,
            'score_layout':self.score_layout,'feature_layout':self.feature_layout,
            'score_definition':'clipped linear endpoint proxy, not a calibrated probability',
            'input_feature_version':cfg['model']['input_feature_version']}

    def train(self,mode=True):
        super().train(False);return self

    @torch.no_grad()
    def encode_features(self,xyz):
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not bool(torch.isfinite(xyz).all()):raise ValueError('Finite partial XYZ only')
        with torch.autocast(xyz.device.type,dtype=torch.float16,enabled=xyz.device.type=='cuda'):
            pyramid=self.model.encode_condition(xyz)
        self.encoder_forward_count+=1;self.last_feature_shapes=tuple(tuple(t.shape) for t in pyramid)
        if tuple(tuple(t.shape) for t in pyramid)!=self.feature_shapes:raise ValueError('Stage1 pyramid geometry/layout mismatch')
        return tuple(t.detach() for t in pyramid)

    @torch.no_grad()
    def forward(self,partial_lidar,*,nfe=4,seed,with_scores=True):
        if nfe not in (1,2,4,8) or not isinstance(seed,int):raise ValueError('Explicit Stage1 NFE and seed required')
        features=self.encode_features(partial_lidar)
        if not with_scores:return features,None
        generator=torch.Generator(device=partial_lidar.device).manual_seed(seed)
        with torch.autocast(partial_lidar.device.type,dtype=torch.float16,enabled=partial_lidar.device.type=='cuda'):
            score,state=self.model.sample(partial_lidar,nfe=nfe,generator=generator,condition=features)
        if not bool(torch.isfinite(state).all() and torch.isfinite(score).all()):raise ValueError('Nonfinite Stage1 occupancy')
        return features,score[0,0].detach().float()
