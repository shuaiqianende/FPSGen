import json
import pytest
from fpsgen.hier_occ_flow.storage import file_sha
from fpsgen.hier_occ_flow.verified_adaptation_data import VerifiedAdaptationDataset


def fixture(tmp_path):
    class Dataset:
        root=tmp_path
        index=[{'shard':0}]
        def __len__(self):return 1
        def __getitem__(self,i):return 'validated record'
    for name in ('ids','masks'):(tmp_path/name).write_bytes(b'valid')
    dataset=Dataset();dataset.meta={'shards':[dict(ids='ids',masks='masks',
        ids_sha256=file_sha(tmp_path/'ids'),masks_sha256=file_sha(tmp_path/'masks'))]}
    receipt=tmp_path/'audit.json';receipt.write_text(json.dumps({'status':'PASS'}))
    return VerifiedAdaptationDataset(dataset,receipt),receipt


def test_consumed_shard_is_checked_again_when_changed(tmp_path):
    ds,receipt=fixture(tmp_path)
    assert ds[0]=='validated record'
    (tmp_path/'ids').write_bytes(b'bad!!')
    with pytest.raises(ValueError,match='checksum'):ds[0]


def test_failed_full_audit_never_feeds_a_training_record(tmp_path):
    ds,receipt=fixture(tmp_path)
    receipt.write_text(json.dumps({'status':'FAILED'}))
    with pytest.raises(ValueError,match='Completed'):ds[0]
