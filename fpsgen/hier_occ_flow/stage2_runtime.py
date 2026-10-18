"""Strict384 configuration and dependency identities shared by all new entries."""
import hashlib
import json
from pathlib import Path
import yaml
from .geometry import SparseHierarchyGeometry
from .storage import SparseShardDataset,file_sha,DATA_VERSION


def load_stage2_config(path):
    path=Path(path);cfg=yaml.safe_load(path.read_text())
    cfg['geometry']=yaml.safe_load((path.parent/cfg['geometry']).read_text()) if isinstance(cfg['geometry'],str) else cfg['geometry']
    geometry=SparseHierarchyGeometry(**cfg['geometry'])
    if geometry.parent_resolution!=(384,384,32) or geometry.child_split!=(2,2,2) or geometry.origin!=(-50.,-50.,-4.) or geometry.maximum!=(50.,50.,4.4) or geometry.sphere_radius!=50.:
        raise ValueError('Stage2 formal384+8 geometry is fixed')
    m=cfg['model']
    if m['backbone']!='ptv3' or m['width']!=64 or m['lidar_channels']!=384 or m['patch_sizes']!=[64,128]:
        raise ValueError('Keep the registered64D two-level PTv3 capacity')
    if m['condition_mode']!='pyramid_observed':raise ValueError('Main384 configs require actual LiDAR pyramid plus input child hits')
    if cfg['loss']!={m['objective']:1.} or m['objective'] not in ('fm','bce'):raise ValueError('Only pureFM or matchedBCE')
    if cfg['flow']!={'source':'gaussian','sigma':1.,'target_mapping':'binary_symmetric'}:raise ValueError('Binary occupancy flow only')
    if cfg['training']['successful_updates']>100000 or cfg['training']['batch_per_gpu'] not in (1,2):raise ValueError('Register<=100K updates, one or two full scenes/rank')
    if cfg['training']['batch_per_gpu']==2 and cfg['training'].get('batching')!='packed_variable_length':raise ValueError('B2 requires actual packed variable-length scenes')
    if cfg['training']['precision'] not in ('fp16','bf16','fp32'):raise ValueError('Unknown precision')
    if cfg['training']['scope'] not in ('engineering_smoke','overfit','pilot1000','pilot5000','formal'):raise ValueError('Explicit experimental phase required')
    if cfg['training']['scope']=='engineering_smoke' and cfg['training']['successful_updates']>32:raise ValueError('Engineering smoke<=32 updates')
    if cfg['candidates']['mode'] not in ('oracle','predicted'):raise ValueError('Explicit support source required')
    if cfg.get('research_protocol')=='fm_real_stage1_score_gt_parents_only_v1':
        c=cfg['candidates']
        if c['mode']!='oracle' or c.get('oracle_score')!='stage1' or c.get('negative_fraction')!=0:
            raise ValueError('This variant requires actual Stage1 scores and GT parents only')
    if cfg['stage1']['type']!='occupancy384':raise ValueError('Old density adapters are prohibited')
    if cfg['stage1']['checkpoint_sha256']!=cfg['stage1']['adapter']['checkpoint_sha256']:
        raise ValueError('Stage1 adapter/checkpoint/gate identity must agree')
    if cfg['validation']['nfes']!=[1,2,4,8]:raise ValueError('Keep the registered1/2/4/8 NFE evaluation')
    if cfg['candidates'].get('stage1_nfe') not in (1,2,4,8):raise ValueError('Invalid frozen Stage1 NFE')
    if cfg['training']['max_parent_tokens']<=0:raise ValueError('Positive explicit token capacity required')
    if cfg['candidates']['mode']=='predicted' and cfg['training']['scope']=='formal' and cfg['candidates'].get('calibration_status')!='frozen_development':
        raise ValueError('Formal predicted support must be calibrated and frozen on development')
    return cfg,geometry


def dataset_identity(cfg,g,*,verify=False):
    from .morphology import MORPH_DATA_VERSION
    version=cfg.get('target_data_version',DATA_VERSION)
    if version not in (DATA_VERSION,MORPH_DATA_VERSION):raise ValueError('Unknown fine target version')
    train=SparseShardDataset(cfg['train_dataset'],g,verify=verify,data_version=version)
    dev=SparseShardDataset(cfg['development_dataset'],g,verify=verify,data_version=version)
    if version==MORPH_DATA_VERSION:
        if any(not isinstance(cfg.get(key),str) or len(cfg[key])!=64 for key in ('target_transform_sha256','target_processor_sha256')):
            raise ValueError('Explicit morphology transform/processor identity required')
        for name,ds in (('train',train),('development',dev)):
            raw=cfg.get('raw_target_datasets',{}).get(name)
            if (not raw or ds.meta.get('source_parent_manifest_sha256')!=file_sha(Path(raw)/'manifest.json')
                or ds.meta.get('target_transform_sha256')!=cfg.get('target_transform_sha256')
                or ds.meta.get('target_processor_sha256')!=cfg['target_processor_sha256']
                or ds.meta.get('parent_ids_unchanged') is not True):
                raise ValueError('Morphology target source/transform identity mismatch')
    keys=lambda ds:{(r['sequence'].decode(),int(r['frame'])) for r in ds.index}
    a,b=keys(train),keys(dev)
    if a&b or any(s=='08' for s,f in a|b):raise ValueError('Stage2 train/development must not overlap or contain seq08')
    if len(train)!=17870 or len(dev)!=900:raise ValueError('Full fixed384 splits required before cohort subsetting')
    return train,dev,{'train_manifest_sha256':file_sha(Path(cfg['train_dataset'])/'manifest.json'),
        'development_manifest_sha256':file_sha(Path(cfg['development_dataset'])/'manifest.json')}


def config_sha(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def source_sha():
    """Hash actual model/trainer/vendor sources even before a Git commit exists."""
    root=Path(__file__).resolve().parents[2];digest=hashlib.sha256()
    folders=('fpsgen/hier_occ_flow','fpsgen/voxel_point/vendor/pwm/ptv3_reference',
             'scripts/research_v2/hier_occ_flow')
    for folder in folders:
        for path in sorted((root/folder).rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def sampler_index(attempt,rank,world,length,seed):
    position=attempt*world+rank;epoch,index=divmod(position,length)
    return int(__import__('numpy').random.default_rng(seed+epoch).permutation(length)[index])


def frame_budget_plan(successful_updates,successful_frames,target_frames,global_batch):
    """Count successful scene exposures, including earlier variable-batch phases."""
    if any(type(x) is not int or x<0 for x in (successful_updates,successful_frames,target_frames)) or global_batch<=0:
        raise ValueError('Nonnegative integer successful counters and positive global batch required')
    remaining=target_frames-successful_frames
    if remaining<0 or remaining%global_batch:
        raise ValueError('Exact frame budget must be reachable without dropping a scene from the global batch')
    return {'target_successful_frame_exposures':target_frames,'inherited_successful_frame_exposures':successful_frames,
            'remaining_successful_updates':remaining//global_batch,
            'target_successful_updates':successful_updates+remaining//global_batch,'global_batch':global_batch}


def direct_full_train_policy(cfg,g):
    """An explicit user research decision may waive QUALITY gates, never engineering."""
    policy=cfg.get('direct_training')
    if policy is None:return None
    if (policy.get('authorization')!='user_requested_100k_successful_frames'
        or policy.get('quality_gate_policy')!='monitor_only'
        or cfg['training']['scope']!='formal' or cfg.get('cohort_indices') is not None
        or cfg['training'].get('successful_frame_exposures')!=100000
        or cfg['training']['batch_per_gpu']!=2 or cfg['model']['objective']!='fm'
        or cfg.get('research_protocol')!='fm_only_user_requested' or cfg['candidates']['mode']!='oracle'):
        raise ValueError('Direct100K-frame authorization is specific to the full-split Oracle FM GB4 experiment')
    path=Path(policy['previous_quality_evidence'])
    if file_sha(path)!=policy['previous_quality_sha256']:raise ValueError('Preserve the real prior quality evidence')
    evidence=json.loads(path.read_text())
    if (evidence.get('status') not in ('FAIL_ORACLE_OVERFIT','PASS_ORACLE_OVERFIT')
        or evidence.get('geometry_sha256')!=g.sha256
        or evidence.get('stage1_checkpoint_sha256')!=cfg['stage1']['checkpoint_sha256']):
        raise ValueError('Direct-training quality evidence identity mismatch')
    return {**policy,'actual_previous_quality_status':evidence['status'],
            'quality_pass_claimed':False,'engineering_checks_waived':False}


def require_training_gate(cfg,g):
    scope=cfg['training']['scope'];path=cfg.get('engineering_gate')
    gate=json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
    required=['full_scene_gpu','maximum_scene_gpu','ptv3_permutation','checkpointing_parity']
    if scope!='engineering_smoke':required+=['ddp_resume']
    if cfg['training']['batch_per_gpu']==2:
        required+=['packed_batch2_gpu']
        if scope!='engineering_smoke':required+=['packed_batch2_ddp_resume']
    fm_only=cfg.get('research_protocol')=='fm_only_user_requested' and cfg['model']['objective']=='fm'
    direct=direct_full_train_policy(cfg,g)
    if not direct:
        if scope not in ('overfit','engineering_smoke'):required+=['oracle_fm_overfit'] if fm_only else ['oracle_bce_overfit','oracle_fm_overfit']
        if scope in ('pilot5000','formal'):required+=['fm_pilot_1k'] if fm_only else ['matched_bce_fm_1k']
        if scope=='formal':required+=['pilot5000','stage1_final_checkpoint','development_candidate_calibration','formal_budget_registered']
    if gate.get('geometry_sha256')!=g.sha256 or any(gate.get(k) is not True for k in required):
        raise ValueError(f'Stage2{scope} gate missing actual evidence: {[k for k in required if gate.get(k) is not True]}')
    if gate.get('stage1_checkpoint_sha256')!=cfg['stage1']['checkpoint_sha256']:raise ValueError('Gate Stage1 identity differs')
    return gate
