"""Independent pretrained-EMA adaptation; original scratch/frozen paths stay intact."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .condition import sample_pyramid
from .stage1_occ_adapter import FrozenStage1OccupancyAdapter,validate_stage1_identity
from .stage1_occupancy import occupancy_flow_pair,masked_occ_loss,ids_to_dense_occupancy
from .stage2_model import Stage2OccupancyModel
from .stage2_dataset import assemble_stage2_scene
from .stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from .stage1_training import state_sha
from .storage import file_sha

STAGE1_SHA='b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2'
STAGE2_SHA='eaf70f8e2cbc068add3b744b1988299197c787b8f15567fd19fd06b0d5f3f74c'
ARMS=('EF1','EF2','S2-E0','S2-E1','S2-E2')


def parameter_sha(module):
    h=hashlib.sha256()
    for name,value in sorted(module.named_parameters()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_state_sha(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_prefixes(arm):
    return {'EF1':('stage1.encoder.',),'EF2':('stage1.encoder.','stage1.flow.'),
        'S2-E0':('stage2.',),'S2-E1':('stage1.encoder.',),
        'S2-E2':('stage1.encoder.','stage2.')}[arm]


def frozen_state_sha(system):
    return tensor_state_sha({k:v for k,v in system.state_dict().items() if not k.startswith(train_prefixes(system.arm))})


def update_adaptation_ema(ema,online,decay):
    """Average only registered trainable modules, preserving frozen state bitwise."""
    source=online.state_dict();prefixes=train_prefixes(online.arm)
    with torch.no_grad():
        for key,target in ema.state_dict().items():
            if not key.startswith(prefixes):continue
            if target.is_floating_point():target.mul_(decay).add_(source[key],alpha=1-decay)
            else:target.copy_(source[key])


class EncoderAdaptationSystem(nn.Module):
    def __init__(self,stage1,arm,stage2=None):
        super().__init__()
        if arm not in ARMS or (arm.startswith('S2-') and stage2 is None):raise ValueError('Registered adaptation arm required')
        self.stage1=stage1;self.stage2=stage2;self.arm=arm
        self.requires_grad_(False)
        for name,p in self.named_parameters():p.requires_grad_(name.startswith(train_prefixes(arm)))
        self.train(True)

    def train(self,mode=True):
        super().train(mode)
        if not mode:return self
        self.stage1.encoder.train(self.arm!='S2-E0')
        self.stage1.flow.train(self.arm=='EF2')
        if self.stage2 is not None:
            self.stage2.train(self.arm in ('S2-E0','S2-E2'))
            if self.arm=='S2-E1':
                # Enable only the existing non-vendor checkpoint wrappers.
                # Children stay in eval (no mutable stats/dropout). Frozen
                # weights do not preclude feature-input gradients.
                for block in self.stage2.modules():
                    if type(block).__name__=='Checkpointed':block.training=True
        return self

    def optimizer_groups(self,training):
        groups=[]
        for prefix,lr in (('stage1.encoder.',training['lr_encoder']),('stage1.flow.',training['lr_flow']),('stage2.',training['lr_stage2'])):
            params=[p for name,p in self.named_parameters() if name.startswith(prefix) and p.requires_grad]
            if params:groups.append({'params':params,'lr':lr,'name':prefix[:-1]})
        if not groups:raise ValueError('No registered trainable parameters')
        return groups

    def trainability_audit(self):
        rows=[{'name':name,'trainable':p.requires_grad,'shape':list(p.shape),'dtype':str(p.dtype)} for name,p in self.named_parameters()]
        if any(r['trainable']!=r['name'].startswith(train_prefixes(self.arm)) for r in rows):raise ValueError('Freeze policy mismatch')
        return rows

    def stage1_loss(self,xyz,targets,generator):
        pair=occupancy_flow_pair(targets,self.stage1.valid,generator=generator)
        pyramid=self.stage1.encoder(xyz)
        observed=torch.cat([self.stage1.geometry.observed_mask(x) for x in xyz])
        def velocity(state,time,*features):return self.stage1.flow(state,observed,features,time)*self.stage1.valid
        # Whole frozen Flow recomputation retains d(loss)/d(condition) while
        # keeping BN/other frozen module state in eval. Never no_grad here.
        prediction=checkpoint(velocity,pair['state'],pair['time'],*pyramid,use_reentrant=False)
        loss=masked_occ_loss(prediction,pair['velocity'],self.stage1.valid,objective='fm')
        return loss,{'pair':pair,'pyramid':pyramid}


def load_sources(cfg,device):
    """Strict EMA loading and source lineage audit, without invoking scratch runners."""
    spec=cfg['source_stage1']
    if spec['checkpoint_sha256']!=STAGE1_SHA or file_sha(spec['checkpoint'])!=STAGE1_SHA:
        raise ValueError('Exact registered Stage1 checkpoint required; no substitute')
    adapter=FrozenStage1OccupancyAdapter(**spec)
    stage1=adapter.model;g=adapter.geometry
    stage1.encoder.activation_checkpointing=True
    stage1.flow.checkpointing=False  # Outer complete-Flow checkpoint, not nested recomputation.
    initial={'initialization_mode':'pretrained_ema_adaptation','stage1_checkpoint_sha256':STAGE1_SHA,
        'stage1_source_provenance':copy.deepcopy(adapter.provenance),
        'stage1_initial_state_sha256':state_sha(stage1),'encoder_initial_parameter_sha256':parameter_sha(stage1.encoder),
        'flow_initial_state_sha256':state_sha(stage1.flow),'stage1_loaded_tensors':[
            {'name':k,'shape':list(v.shape),'dtype':str(v.dtype),'loaded':True} for k,v in stage1.state_dict().items()],
        'ema_initialization':'fresh copy of strictly loaded source','optimizer_initialization':'new empty state'}
    stage2=None
    if cfg['arm'].startswith('S2-'):
        stage2,audit=load_stage2_source(cfg,g);initial['stage2']=audit
    model=EncoderAdaptationSystem(stage1,cfg['arm'],stage2).to(device)
    initial['trainability']=model.trainability_audit();initial['frozen_state_sha256']=frozen_state_sha(model)
    return model,g,initial


def load_stage2_source(cfg,g):
    path=Path(cfg['source_stage2']['checkpoint'])
    if cfg['source_stage2']['checkpoint_sha256']!=STAGE2_SHA or file_sha(path)!=STAGE2_SHA:
        raise ValueError('Exact registered original Stage2 source required')
    value=torch.load(path,map_location='cpu');identity=value['identity']
    source_cfg=json.loads((Path(identity['output'])/'config.json').read_text())
    from .stage2_runtime import config_sha
    if (identity['geometry_sha256']!=g.sha256 or identity['config_sha256']!=config_sha(source_cfg)
        or identity['stage1_checkpoint_sha256']!=STAGE1_SHA or not value.get('parameter_sync_pass')
        or source_cfg.get('target_data_version','hs-occ-surface-hit-v1')!='hs-occ-surface-hit-v1'):
        raise ValueError('Stage2 raw-GT/source identity mismatch')
    for split in ('train','development'):
        if file_sha(Path(cfg[split+'_dataset'])/'manifest.json')!=identity[split+'_manifest_sha256']:
            raise ValueError('Stage2 source uses different data')
    options=copy.deepcopy(source_cfg['model'])
    if options.get('context_strategy','dynamic')!='dynamic' or any(options.get(k,'none')!='none' for k in ('child_bridge','state_condition_interaction','local_state_encoding')):
        raise ValueError('Only original Dynamic FM without experimental branches')
    model=Stage2OccupancyModel(g,**options);model.load_state_dict(value['model'],strict=True)
    return model,{'checkpoint_sha256':STAGE2_SHA,'weights':'online','source_identity':identity,
        'initial_parameter_sha256':parameter_sha(model),'loaded_tensors':list(model.state_dict()),
        'source_successful_exposures':value['sampler']['successful_frame_exposures']}


def stage2_differentiable_batch(system,records,xyz,cfg,g,teacher,*,teacher_scenes=None):
    """B2 Encoder -> differentiable 3D sampling -> packed PTv3. No feature cache."""
    if teacher_scenes is None:
        policy=json.loads(Path(cfg['reference_stage2_config']).read_text())
        teacher_scenes=[assemble_stage2_scene(r,policy,g,xyz[0].device,teacher,cache=None) for r in records]
    # Discrete teacher identities/scores/masks are detached. Their old frozen
    # feature tensor is discarded, never used as the trainable condition.
    discrete=[(ids,{k:v for k,v in condition.items() if k!='lidar_features'},target,provenance)
              for ids,condition,target,provenance in teacher_scenes]
    del teacher_scenes
    pyramid=system.stage1.encoder(xyz)  # One real batch2 dense forward.
    scenes=[]
    for index,(ids,condition,target,provenance) in enumerate(discrete):
        centers=g.centers(g.unflatten(ids)).float()
        features=sample_pyramid(tuple(p[index:index+1] for p in pyramid),centers,g.origin,g.maximum).half()
        if system.stage1.encoder.training and not features.requires_grad:raise RuntimeError('Encoder-to-grid_sample gradient path was cut')
        condition['lidar_features']=features
        scenes.append((ids,condition,target,provenance))
    ids,condition,target,lengths=pack_scenes(scenes)
    return ids,condition,target,lengths,pyramid


def stage2_guided_loss(system,records,xyz,cfg,g,teacher,generator):
    ids,condition,target,lengths,pyramid=stage2_differentiable_batch(system,records,xyz,cfg,g,teacher)
    pair=packed_training_pair(target,condition['valid_child_mask'],condition['batch'],generator)
    velocity=system.stage2(ids,state=pair['state'],time=pair['time'],**condition)
    return scene_mean_mse(velocity,pair['velocity'],condition['valid_child_mask'],condition['batch']),{
        'ids':ids,'condition':condition,'target':target,'pair':pair,'scene_lengths':lengths,'pyramid':pyramid}
