"""Reuse completed full audit, then checksum every consumed shard before access."""
import json
import time
from pathlib import Path
from .storage import file_sha


class VerifiedAdaptationDataset:
    def __init__(self, dataset, receipt):
        self.dataset=dataset;self.receipt=Path(receipt);self.verified={}

    def __getattr__(self,name):return getattr(self.dataset,name)
    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        if json.loads(self.receipt.read_text())['status']!='PASS':
            raise ValueError('Completed full dataset audit is required')
        shard=self.dataset.meta['shards'][int(self.dataset.index[index]['shard'])]
        for field in ('ids','masks'):
            path=self.dataset.root/shard[field];st=path.stat()
            signature=(st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino)
            # Some shared filesystems have coarse timestamps. Never reuse a
            # checksum for recently written files: same-size edits in one clock
            # tick must be read and hashed again. Frozen historical shards are
            # eligible for the normal metadata-keyed cache.
            recent=time.time_ns()-max(st.st_mtime_ns,st.st_ctime_ns)<2_000_000_000
            if recent or self.verified.get(str(path))!=signature:
                if file_sha(path)!=shard[field+'_sha256']:raise ValueError('Adaptation shard checksum failed')
                after=path.stat()
                if (after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino)!=signature:
                    raise ValueError('Shard changed during verification')
                self.verified[str(path)]=signature
        return self.dataset[index]


def verify_completed_audit(train,dev,g,manifests,receipt):
    audit=json.loads(Path(receipt).read_text())
    if (audit.get('status')!='PASS' or not audit.get('all_original_content_checks_passed')
            or audit['geometry_sha256']!=g.sha256 or audit['manifests']!=manifests):
        raise ValueError('Full audit belongs to different or unverified data')
    for name,ds in [('train',train),('development',dev)]:
        if str(ds.root.resolve())!=audit['roots'][name]:raise ValueError('Full audit root mismatch')
        if file_sha(ds.root/ds.meta['index'])!=ds.meta['index_sha256']:raise ValueError('Index checksum failed')
    return VerifiedAdaptationDataset(train,receipt),VerifiedAdaptationDataset(dev,receipt)
