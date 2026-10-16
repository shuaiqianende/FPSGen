"""Actual parent-ID plus partial-XYZ payload through DataLoader0/2/4/8 workers.

Only warm-cache throughput is claimed; never flush the server's global cache.
The actual trainer additionally records CPU input time vs GPU update time.
"""
from train_stage1_occ import load_config
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json
from fpsgen.hier_occ_flow.dataset import input_xyz


class Samples(Dataset):
    def __init__(self,root,geometry,data_root,count):
        self.source=SparseShardDataset(root,geometry);self.data_root=Path(data_root)
        self.indices=np.random.default_rng(20261017).integers(len(self.source),size=count)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        row=self.source[int(self.indices[i])]
        xyz=input_xyz(self.data_root/row['sequence']/'input_'/f"{row['frame']:06d}.npy")
        # PyTorch2.0 cannot ingest uint32; the same conversion as training.
        return torch.from_numpy(row['parent_ids'].astype(np.int64)),torch.from_numpy(xyz)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=128)
    args=p.parse_args();cfg=load_config(args.config);g=SparseHierarchyGeometry(**cfg['geometry']);torch.set_num_threads(1)
    rows=[]
    for workers in (0,2,4,8):
        ds=Samples(cfg['train_dataset'],g,cfg['data_root'],args.samples)
        # Warm the *same entire fixture* for every worker setting. Warming only
        #16/128 makes the first configuration pay cold reads for the other112.
        for index in range(len(ds)):ds[index]
        kwargs={'persistent_workers':True,'prefetch_factor':2} if workers else {}
        loader=DataLoader(ds,batch_size=1,num_workers=workers,shuffle=False,pin_memory=False,**kwargs)
        start=time.perf_counter();payload=0;checksum=0
        for ids,xyz in loader:
            payload+=ids.numel()*ids.element_size()+xyz.numel()*xyz.element_size()
            checksum+=int(ids.sum())+float(xyz.sum())
        elapsed=time.perf_counter()-start
        rows.append({'workers':workers,'frames':len(ds),'warmup_frames':len(ds),'seconds':elapsed,'frames_per_second':len(ds)/elapsed,
            'ipc_payload_bytes':payload,'ipc_payload_bytes_per_second':payload/elapsed,'checksum':checksum})
        del loader,ds
    atomic_json(args.output,{'scope':'full train dataset,128 seeded random frames, warmed cache; startup included',
        'cold_cache_measured':False,'gpu_wait_measured':False,'rows':rows})
    print(rows)

if __name__=='__main__':main()
