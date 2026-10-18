"""Compare6/26-neighbor fine closing on frozen train/development representatives."""
from common import ROOT
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.morphology import FineClosingSpec,close_fine_masks
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sparsity-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg,g=load_stage2_config(a.config);train,dev,identity=dataset_identity(cfg,g)
    source=json.loads(a.sparsity_audit.read_text())
    if source['status']!='COMPLETE' or source['data_identity']!=identity:raise ValueError('Full split audit identity differs')
    report={'status':'RUNNING','geometry_sha256':g.sha256,'data_identity':identity,
        'source_sparsity_audit_sha256':file_sha(a.sparsity_audit),'test_frames_used':False,'rows':[]}
    for name,ds in (('train',train),('development',dev)):
        for representative in source['splits'][name]['representatives']:
            record=ds[representative['index']];ids=record['parent_ids'];masks=record['child_occ_mask']
            centers=g.centers(occupied_fine_coords(ids,masks,g),fine=True);tree=cKDTree(centers,leafsize=32)
            for connectivity in (6,26):
                spec=FineClosingSpec(connectivity=connectivity);start=time.perf_counter()
                new,stats=close_fine_masks(ids,masks,g,spec)
                added=g.centers(occupied_fine_coords(ids,new^masks,g),fine=True)
                distance=tree.query(added,k=1,eps=0,workers=4)[0] if len(added) else np.zeros(1)
                row={'split':name,'sequence':record['sequence'],'frame':record['frame'],'quantile':representative['quantile'],
                    'transform':spec.to_dict(),**stats,'seconds':time.perf_counter()-start,
                    'added_center_to_original_center_distance_m':{str(q):float(np.percentile(distance,q)) for q in (50,90,99,100)}}
                report['rows'].append(row);atomic_json(a.output,report)
                print(f'{name}/{record["sequence"]}/{record["frame"]:06d} C{connectivity}: +{stats["added_fraction"]:.2%}, {row["seconds"]:.2f}s',flush=True)
    report['status']='COMPLETE';atomic_json(a.output,report)


if __name__=='__main__':main()
