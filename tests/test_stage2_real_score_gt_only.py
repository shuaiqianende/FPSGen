"""CPU contracts for real-score GT-only conditions; no GPU pass is implied."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import yaml
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage2_candidates import oracle_parent_candidates
from fpsgen.hier_occ_flow.stage2_condition import input_conditions
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene,ConditionCache
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,require_training_gate
from fpsgen.hier_occ_flow.occupancy_sparsity import mask_histogram,summarize_histogram
from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel


@pytest.fixture
def g():return SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))


def parents(g):
    return g.flatten(np.array([[180,190,16],[181,190,16],[182,190,16]]))


class ScoreAdapter:
    provenance={'type':'CPU_TEST_STUB','checkpoint_sha256':'test-only'}
    def __init__(self,g,ids):self.g=g;self.ids=ids;self.calls=[]
    def __call__(self,xyz,**kwargs):
        self.calls.append(kwargs)
        score=torch.zeros(self.g.parent_resolution[::-1])
        score.view(-1)[self.ids]=torch.tensor([.07,.91,0.])
        return [torch.ones(1,128,2,2,2) for _ in range(3)],score


def policy():
    return {'mode':'oracle','oracle_score':'stage1','negative_fraction':0.,'stage1_nfe':4,'seed':20261018}


def test_gt_only_never_expands_or_draws_empty_parents(g):
    ids=parents(g)
    with patch('fpsgen.hier_occ_flow.stage2_candidates.expand_parent_ids',side_effect=AssertionError('No expansion')):
        selected,_,meta=oracle_parent_candidates(ids[[2,0,0,1]],g,seed=4,negative_fraction=0.)
    np.testing.assert_array_equal(selected,ids)
    assert meta['nearby_negatives']==0 and meta['nearby_pool_evaluated'] is False
    assert meta['negative_fraction_relative_to_gt']==0


def test_actual_score_replaces_constant1_without_thresholding_gt_support(g):
    ids=parents(g);adapter=ScoreAdapter(g,ids)
    xyz=torch.from_numpy(g.centers(g.unflatten(ids[:1])).astype(np.float32))
    selected,c,source=input_conditions(xyz,g,adapter,policy(),seed=3,oracle_ids=ids)
    np.testing.assert_array_equal(selected.numpy(),ids)
    torch.testing.assert_close(c['parent_score'],torch.tensor([.07,.91,0.]))
    assert adapter.calls==[{'nfe':4,'seed':3,'with_scores':True}]
    assert source['parent_score_source']=='actual_stage1_endpoint'
    assert source['nearby_negatives']==0 and not source['stage1_score_is_calibrated_probability']
    assert c['observed_child'].sum()==1
    assert c['lidar_features'].shape==(3,384)
    assert all(not value.requires_grad for value in c.values())


def test_child_gt_can_still_be_empty_inside_occupied_parent_without_condition_leak(g,tmp_path):
    ids=parents(g);directory=tmp_path/'00/input_';directory.mkdir(parents=True)
    np.save(directory/'000000.npy',np.column_stack((g.centers(g.unflatten(ids[:1])),[99.])))
    cfg={'data_root':str(tmp_path),'candidates':policy(),'training':{'max_parent_tokens':100}}
    record={'sequence':'00','frame':0,'parent_ids':ids.astype(np.uint32),
        'child_occ_mask':np.array([1,128,255],np.uint8)}
    a=assemble_stage2_scene(record,cfg,g,torch.device('cpu'),ScoreAdapter(g,ids))
    b=assemble_stage2_scene({**record,'child_occ_mask':np.array([3,4,127],np.uint8)},cfg,g,torch.device('cpu'),ScoreAdapter(g,ids))
    assert len(a[0])==3 and bool(a[2].any(1).all()) and bool((~a[2]).any())
    assert all(torch.equal(a[1][k],b[1][k]) for k in a[1]) and not torch.equal(a[2],b[2])
    old={**policy(),'oracle_score':'neutral','negative_fraction':.25}
    assert ConditionCache.key(old)!=ConditionCache.key(policy())


def test_score_has_nonzero_gradient_in_actual_fusion_mlp(g):
    torch.manual_seed(2);ids=torch.from_numpy(parents(g))
    model=HierarchicalOccupancyModel(g,backbone='local')
    scores=torch.tensor([.07,.91,.2],requires_grad=True)
    values=model(ids,state=torch.randn(3,8),time=torch.tensor(.4),parent_score=scores,
        lidar_features=torch.randn(3,384),observed_child=torch.zeros(3,8),
        valid_child_mask=torch.tensor(g.valid_children(ids.numpy())))
    values.square().mean().backward()
    assert scores.grad is not None and bool(torch.isfinite(scores.grad).all()) and float(scores.grad.abs().sum())>0


def test_new_configs_enforce_real_score_gt_only_and_require_new_evidence(tmp_path):
    for name in ('overfit','smoke'):
        path=Path(f'configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_{name}.yaml')
        cfg,g=load_stage2_config(path)
        assert cfg['candidates']['oracle_score']=='stage1' and cfg['candidates']['negative_fraction']==0.
        assert cfg['training']['batch_per_gpu']==2 and not cfg['cache']['enabled']
        with pytest.raises(ValueError,match='gate missing'):require_training_gate(cfg,g)
        cfg['candidates']['oracle_score']='neutral';cfg['geometry']=g.to_dict()
        bad=tmp_path/'bad.yaml';bad.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError,match='actual Stage1 scores'):load_stage2_config(bad)


def test_child_surface_fraction_is_not_parent_fill_all_requirement():
    result=summarize_histogram(mask_histogram(np.array([1,3,15,255],np.uint8)))
    assert result['occupied_fine_children']==15 and result['occupied_parents']==4
    assert result['occupied_children_per_parent_histogram']==[0,1,1,0,1,0,0,0,1]
    assert result['fine_fill_fraction_inside_occupied_parents_nominal8']==15/32
    assert result['all8_children_occupied_parent_fraction']==.25
    for masks in (np.array([0],np.uint8),np.array([256]),np.array([-1])):
        with pytest.raises(ValueError):mask_histogram(masks)
    with pytest.raises(ValueError):summarize_histogram(np.zeros(256,dtype=np.int64))


def test_morphology_dataset_identity_requires_explicit_source_and_transform(g,tmp_path):
    from fpsgen.hier_occ_flow.stage2_runtime import dataset_identity
    from fpsgen.hier_occ_flow.storage import file_sha,INDEX_DTYPE
    from fpsgen.hier_occ_flow.morphology import MORPH_DATA_VERSION,FineClosingSpec
    from fpsgen.hier_occ_flow.morphology_storage import processor_sha
    cfg={'target_data_version':MORPH_DATA_VERSION,'target_transform_sha256':FineClosingSpec().sha256,
         'target_processor_sha256':processor_sha(),'raw_target_datasets':{}}
    sources={}
    class FakeDataset:
        def __init__(self,root,geometry,*,verify,data_version):
            assert geometry==g and data_version==MORPH_DATA_VERSION
            name=Path(root).name;self.meta=sources[name]
            length=17870 if name=='train' else 900
            self.index=np.zeros(length,dtype=INDEX_DTYPE)
            self.index['sequence']=b'00'
            self.index['frame']=np.arange(length)+(0 if name=='train' else 20000)
        def __len__(self):return len(self.index)
    for name in ('train','development'):
        raw=tmp_path/f'raw_{name}';raw.mkdir();(raw/'manifest.json').write_text('{}')
        target=tmp_path/name;target.mkdir();(target/'manifest.json').write_text('{}')
        cfg[f'{name}_dataset']=str(target);cfg['raw_target_datasets'][name]=str(raw)
        sources[name]={'source_parent_manifest_sha256':file_sha(raw/'manifest.json'),
            'target_transform_sha256':cfg['target_transform_sha256'],
            'target_processor_sha256':cfg['target_processor_sha256'],'parent_ids_unchanged':True}
    with patch('fpsgen.hier_occ_flow.stage2_runtime.SparseShardDataset',FakeDataset):
        a,b,_=dataset_identity(cfg,g);assert len(a)==17870 and len(b)==900
        cfg['target_processor_sha256']='0'*64
        with pytest.raises(ValueError,match='source/transform identity'):dataset_identity(cfg,g)
