"""CPU full-training token audit, exact25% oracle sizes with proved empty pools."""
from common import ROOT
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.stage2_candidates import parent_valid_field,oracle_parent_candidates
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry

_DS=None;_G=None;_VALID=None


def initialize(root,geometry):
    global _DS,_G,_VALID
    _G=SparseHierarchyGeometry(**geometry);_DS=SparseShardDataset(root,_G);_VALID=parent_valid_field(_G).ravel()


def measure(index):
    record=_DS[index];ids=record['parent_ids'].astype(np.int64);plane=_G.parent_resolution[0]*_G.parent_resolution[1]
    # A subset of the27-neighborhood is enough to PROVE availability of25%
    # empty parents. This avoids a full4.7M-cell dilation for every frame.
    neighbors=np.unique(np.concatenate((ids[ids//plane<31]+plane,ids[ids//plane>0]-plane)))
    neighbors=neighbors[_VALID[neighbors]]
    where=np.searchsorted(ids,neighbors);hit=(where<len(ids))&(ids[np.minimum(where,len(ids)-1)]==neighbors)
    lower=int((~hit).sum());needed=int(len(ids)*.25)
    if lower>=needed:n=needed;proof='vertical_neighbors_prove_full_pool_sufficient'
    else:
        _,_,source=oracle_parent_candidates(ids,_G,seed=20261018)
        n=source['nearby_negatives'];proof='exact_full_27_neighborhood_count'
    popcount=np.asarray([bin(i).count('1') for i in range(256)],dtype=np.uint8)
    return {'index':index,'sequence':record['sequence'],'frame':record['frame'],'gt_parents':len(ids),
        'fine_occupied':int(popcount[record['child_occ_mask']].sum()),'oracle_negative_parents':n,
        'oracle_tokens':len(ids)+n,'empty_pool_lower_bound':lower,'count_proof':proof}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();cfg,g=load_stage2_config(args.config);args.output.mkdir(parents=True,exist_ok=True)
    train,dev,identity=dataset_identity(cfg,g,verify=True)
    source_audit=Path(cfg['train_dataset']).parent/'full_data_audit.json';old=json.loads(source_audit.read_text())
    if old['status']!='PASS_FULL_DATA_AUDIT' or old['manifest_sha256']!={'train':identity['train_manifest_sha256'],'development':identity['development_manifest_sha256']}:
        raise ValueError('Existing full packed-label audit is not bound to these verified shard bytes')
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(cfg['train_dataset'],g.to_dict())) as pool:
        for row in pool.map(measure,range(len(train)),chunksize=16):
            rows.append(row)
            if len(rows)%200==0 or len(rows)==len(train):
                atomic_json(args.output/'token_audit_progress.json',{'frames':len(rows),'total':len(train),'seconds':time.perf_counter()-start})
                print(f'Oracle token audit {len(rows)}/{len(train)} seconds={time.perf_counter()-start:.1f}',flush=True)
    lengths=np.asarray([r['oracle_tokens'] for r in rows]);representatives=[]
    for q in (50,90,99,100):
        index=int(np.argmin(abs(lengths-np.percentile(lengths,q))))
        representatives.append({'quantile':q,**rows[index]})
    report={'status':'PASS_CPU_FULL_TRAINING_TOKEN_AUDIT','geometry_sha256':g.sha256,**identity,
        'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],'train_frames':len(train),'development_frames':len(dev),
        'guard_frames':360,'seq08_excluded':True,'data_version':train.meta['data_version'],'source_full_audit_sha256':file_sha(source_audit),
        'oracle_token_quantiles':{str(q):float(np.percentile(lengths,q)) for q in (0,50,90,99,100)},
        'gt_parent_quantiles':{str(q):float(np.percentile([r['gt_parents'] for r in rows],q)) for q in (0,50,90,99,100)},
        'representatives':representatives,'rows':rows,'seconds':time.perf_counter()-start,
        'predicted_token_quantiles':'NOT_MEASURED_REQUIRES_FROZEN_STAGE1_GPU_FORWARD',
        'gpu_forward_backward':'NOT_EXECUTED','proof':'Exact oracle count=min(available27-neighbor empties,floor0.25*GT). Vertical subset lower bound proves sufficiency; exact full neighborhood used otherwise.'}
    atomic_json(args.output/'token_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
