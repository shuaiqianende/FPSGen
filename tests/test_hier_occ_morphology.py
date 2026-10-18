"""Exact tiled/global parity, original-hit preservation and transactional resume."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import binary_closing
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.morphology import FineClosingSpec,close_fine_masks,MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.voxelization import pack_children,unpack_children,occupied_fine_coords
from fpsgen.hier_occ_flow.storage import SparseShardDataset,SparseShardWriter,file_sha,DATA_VERSION
from fpsgen.hier_occ_flow.morphology_storage import derive_dataset


def labels(g):
    # A thin plane with a one-child hole that crosses a parent-tile boundary.
    fine=np.array([[x,y,3] for x in range(2,13) for y in range(2,13) if (x,y)!=(7,7)],np.int64)
    parents,child=g.split_fine(fine);ids=np.unique(g.flatten(parents))
    occ=np.zeros((len(ids),8),bool);occ[np.searchsorted(ids,g.flatten(parents)),child]=True
    return ids.astype(np.uint32),pack_children(occ,g)


@pytest.mark.parametrize('connectivity',[6,26])
def test_tiled_closing_matches_global_and_parent_domain(connectivity):
    g=SparseHierarchyGeometry(parent_resolution=(8,8,4),child_split=(2,2,2),origin=(-2.,-2.,-1.),maximum=(2.,2.,1.))
    ids,masks=labels(g);spec=FineClosingSpec(connectivity=connectivity,tile_parent_xy=2)
    result,report=close_fine_masks(ids,masks,g,spec)
    original=occupied_fine_coords(ids,masks,g);full=np.zeros(g.fine_resolution[::-1],bool)
    full[original[:,2],original[:,1],original[:,0]]=True
    closed=binary_closing(full,structure=spec.footprint,iterations=1)|full
    parent=g.unflatten(ids);query=parent[:,None,:]*2+g.child_offsets()[None]
    expected=closed[query[...,2],query[...,1],query[...,0]]&g.valid_children(ids)
    np.testing.assert_array_equal(result,pack_children(expected,g))
    np.testing.assert_array_equal(result&masks,masks)
    assert report['parent_ids_unchanged'] and report['original_hits_preserved']
    other,_=close_fine_masks(ids,masks,g,FineClosingSpec(connectivity=connectivity,tile_parent_xy=3))
    np.testing.assert_array_equal(other,result)
    if connectivity==26:assert report['added_fine']>0


def test_closing_keeps_boundary_hits_and_no_invalid_children():
    g=SparseHierarchyGeometry(parent_resolution=(8,8,4),child_split=(2,2,2),origin=(-2.,-2.,-1.),maximum=(2.,2.,1.),sphere_radius=1.8)
    all_ids=np.arange(g.num_parents);valid=g.valid_children(all_ids);ids=all_ids[valid.any(1)]
    original=pack_children(g.valid_children(ids),g)
    new,stats=close_fine_masks(ids,original,g,FineClosingSpec(tile_parent_xy=2))
    np.testing.assert_array_equal(new,original)
    assert stats['invalid_children_added']==0


def test_derived_dataset_partial_resume_identity_and_original_immutable(tmp_path):
    g=SparseHierarchyGeometry(parent_resolution=(8,8,4),child_split=(2,2,2),origin=(-2.,-2.,-1.),maximum=(2.,2.,1.))
    source=tmp_path/'source';out=tmp_path/'derived';ids,masks=labels(g)
    writer=SparseShardWriter(source,g,'raw-source-hash',shard_frames=1)
    writer.add('00',0,ids,masks);writer.add('00',1,ids,masks);writer.flush()
    source_sha=file_sha(source/'manifest.json')
    first=derive_dataset(source,out,g,FineClosingSpec(tile_parent_xy=2),workers=1,max_shards=1)
    assert first['status']=='PARTIAL_RESUMABLE' and not (out/'manifest.json').exists()
    with pytest.raises(ValueError,match='identity mismatch'):
        derive_dataset(source,out,g,FineClosingSpec(connectivity=6,tile_parent_xy=2),workers=1,resume=True)
    final=derive_dataset(source,out,g,FineClosingSpec(tile_parent_xy=2),workers=1,resume=True)
    assert final['status']=='COMPLETE' and final['completed_frames']==2
    with pytest.raises(ValueError,match='version mismatch'):SparseShardDataset(out,g)
    ds=SparseShardDataset(out,g,data_version=MORPH_DATA_VERSION,verify=True)
    assert len(ds)==2 and file_sha(source/'manifest.json')==source_sha
    for record in (ds[0],ds[1]):
        np.testing.assert_array_equal(record['parent_ids'],ids)
        np.testing.assert_array_equal(record['child_occ_mask']&masks,masks)
    assert (source/'shard_0000.ids.bin').stat().st_ino==(out/'shard_0000.ids.bin').stat().st_ino
    assert ds.meta['data_version']==MORPH_DATA_VERSION
    from fpsgen.hier_occ_flow.stage2_evaluation import evaluation_ground_truth
    cfg={'target_data_version':MORPH_DATA_VERSION,'train_dataset':str(out),
         'development_dataset':str(tmp_path/'unused'),'raw_target_datasets':{'train':str(source)}}
    primary=evaluation_ground_truth(ds,cfg,g)
    np.testing.assert_array_equal(primary[0]['child_occ_mask'],masks)
    assert not np.array_equal(ds[0]['child_occ_mask'],primary[0]['child_occ_mask'])
    # Corrupt a new mask only; resume must reject it without touching the source.
    p=out/ds.meta['shards'][0]['masks'];data=p.read_bytes();p.write_bytes(bytes([data[0]^1])+data[1:])
    with pytest.raises(ValueError,match='corrupted'):
        derive_dataset(source,out,g,FineClosingSpec(tile_parent_xy=2),workers=1,resume=True)
    assert SparseShardDataset(source,g,verify=True).meta['data_version']==DATA_VERSION


def test_invalid_morphology_options_and_masks_rejected():
    for opts in ({'iterations':2},{'connectivity':18},{'tile_parent_xy':0}):
        with pytest.raises(ValueError):FineClosingSpec(**opts)
