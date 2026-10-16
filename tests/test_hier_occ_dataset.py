import numpy as np
import pytest
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,SparseShardWriter


@pytest.mark.parametrize('split',[(2,2,2),(3,3,2)])
def test_shard_roundtrip_resume_version_corruption(tmp_path,split):
    g=SparseHierarchyGeometry(child_split=split)
    w=SparseShardWriter(tmp_path,g,'test-source',shard_frames=2)
    ids=np.array([0,2,7],dtype=np.uint32); masks=np.array([1,2,4],dtype=g.mask_dtype)
    w.add('00',1,ids,masks); w.add('00',2,ids,masks)
    ds=SparseShardDataset(tmp_path,g,verify=True)
    assert len(ds)==2 and np.array_equal(ds[1]['child_occ_mask'],masks)
    with pytest.raises(FileExistsError): SparseShardWriter(tmp_path,g,'test-source')
    with pytest.raises(ValueError): SparseShardWriter(tmp_path,g,'wrong-source',resume=True)
    w=SparseShardWriter(tmp_path,g,'test-source',resume=True)
    assert not w.add('00',1,ids,masks)
    w.add('05',42,ids,masks); w.flush()
    ds=SparseShardDataset(tmp_path,g,verify=True); assert len(ds)==3 and ds[2]['sequence']=='05'
    (tmp_path/'shard_0000.ids.bin').write_bytes(b'broken')
    with pytest.raises(ValueError): SparseShardDataset(tmp_path,g,verify=True)


def test_uncommitted_shard_is_ignored_on_resume(tmp_path):
    g=SparseHierarchyGeometry(); w=SparseShardWriter(tmp_path,g,'x',shard_frames=1)
    w.add('00',0,np.array([1],np.uint32),np.array([1],np.uint32))
    (tmp_path/'shard_0001.ids.bin').write_bytes(b'partial')
    w=SparseShardWriter(tmp_path,g,'x',resume=True,shard_frames=1)
    w.add('00',1,np.array([2],np.uint32),np.array([2],np.uint32))
    ds=SparseShardDataset(tmp_path,g,verify=True)
    assert len(ds)==2 and ds[1]['parent_ids'].tolist()==[2]
