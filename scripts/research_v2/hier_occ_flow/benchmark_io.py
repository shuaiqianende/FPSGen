"""Real sparse labels: NPZ vs mmap; no global page-cache flush or GPU contention."""
from common import geometry
import argparse
import importlib.util
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json


class RandomReads(torch.utils.data.Dataset):
    def __init__(self,dataset,order,npz_root=None): self.dataset=dataset; self.order=order; self.npz_root=npz_root
    def __len__(self): return len(self.order)
    def __getitem__(self,i):
        index=int(self.order[i])
        if self.npz_root is None:
            v=self.dataset[index]; return v['parent_ids'].astype(np.int64),v['child_occ_mask'].astype(np.int64)
        row=self.dataset.index[index]
        p=self.npz_root/f"{row['sequence'].decode()}_{int(row['frame']):06d}.npz"
        with np.load(p,allow_pickle=False) as a: return a['parent_ids'].astype(np.int64),a['child_occ_mask'].astype(np.int64)


def collate_payload(batch):
    return [(torch.from_numpy(ids),torch.from_numpy(masks)) for ids,masks in batch]


def cpu_snapshot(loader):
    pids=[os.getpid()]
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None: pids.extend(w.pid for w in iterator._workers)
    result={}
    for pid in pids:
        try:
            text=Path(f'/proc/{pid}/stat').read_text(); fields=text[text.rfind(')')+2:].split()
            result[pid]=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        except FileNotFoundError: pass
    return result


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--geometry',required=True,type=Path)
    p.add_argument('--dataset',required=True,type=Path); p.add_argument('--npz',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--reads',type=int,default=128)
    args=p.parse_args(); torch.set_num_threads(4); ds=SparseShardDataset(args.dataset,geometry(args.geometry),verify=True)
    order=np.random.default_rng(42).integers(0,len(ds),args.reads); rows=[]
    for kind in ('npz','mmap'):
        dataset=RandomReads(ds,order,args.npz if kind=='npz' else None)
        for workers in (0,2,4,8):
            loader=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=workers,persistent_workers=workers>0,
                pin_memory=False,collate_fn=collate_payload,**({'prefetch_factor':2,'timeout':30} if workers else {}))
            for iteration in range(2):
                start=time.perf_counter(); before=cpu_snapshot(loader); data_bytes=0; batches=0
                for batch in loader:
                    data_bytes+=sum(ids.numel()*ids.element_size()+masks.numel()*masks.element_size() for ids,masks in batch); batches+=1
                elapsed=time.perf_counter()-start
                after=cpu_snapshot(loader)
                rows.append({'format':kind,'workers':workers,'pass':iteration,'cache_state':'checksum-verification-warmed; includes worker startup' if iteration==0 else 'warm repeated reads',
                    'frames_per_second':len(order)/elapsed,'useful_GB_per_second':data_bytes/elapsed/1e9,'batches_per_second':batches/elapsed,
                    'wall_seconds':elapsed,'main_and_workers_cpu_core_equivalents':sum(value-before.get(pid,0) for pid,value in after.items())/elapsed,'bytes':data_bytes})
            del loader
    actual=sum(p.stat().st_size for p in args.dataset.glob('*') if p.is_file())
    npz_bytes=sum(p.stat().st_size for p in args.npz.glob('*.npz'))
    matching_npz_bytes=sum((args.npz/f"{row['sequence'].decode()}_{int(row['frame']):06d}.npz").stat().st_size for row in ds.index)
    atomic_json(args.output,{'status':'CPU_IO_BENCHMARK','frame_count':len(ds),'reads':len(order),'rows':rows,
        'mmap_bytes':actual,'npz_bytes_all_frames':npz_bytes,'npz_bytes_matching_index_frames':matching_npz_bytes,'compression_support':{x:bool(importlib.util.find_spec(x)) for x in ('zstandard','lz4')},
        'payload':'Actual variable-length parent IDs and packed masks converted to int64 tensors and transferred to main process; useful GB/s counts that expanded training payload',
        'limits':'No privileged/global cache eviction; first pass is NOT a cold-cache benchmark. CPU time sampled from /proc for main and live persistent workers. CPU only; GPU input stall, pinned memory and concurrent-rank throughput are unmeasured.'})
    print(args.output)

if __name__=='__main__': main()
