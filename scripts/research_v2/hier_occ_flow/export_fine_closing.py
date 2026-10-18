"""Export original/augmented sparse fine centers and a local comparison image."""
from common import ROOT
import argparse
import numpy as np
from pathlib import Path
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.morphology import FineClosingSpec,close_fine_masks
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords
from fpsgen.hier_occ_flow.stage2_visualization import write_xyz_ply
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int,default=9511)
    a=p.parse_args();cfg,g=load_stage2_config(a.config)
    train,_,identity=dataset_identity(cfg,g);record=train[a.index]
    spec=FineClosingSpec(connectivity=26)
    ids,old=record['parent_ids'],record['child_occ_mask']
    new,stats=close_fine_masks(ids,old,g,spec)
    original=g.centers(occupied_fine_coords(ids,old,g),fine=True)
    augmented=g.centers(occupied_fine_coords(ids,new,g),fine=True)
    added=g.centers(occupied_fine_coords(ids,new^old,g),fine=True)
    a.output.mkdir(parents=True,exist_ok=False)
    write_xyz_ply(a.output/'original_fine.ply',original,(80,170,230))
    write_xyz_ply(a.output/'closed_fine.ply',augmented,(80,170,230))
    write_xyz_ply(a.output/'added_only.ply',added,(245,100,30))
    import os
    cache=ROOT/'experiments/hier_occ_flow/.matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Choose a fixed physical XY patch with many additions; both panels use
    # the same height slab, all of its original points, and identical axes.
    bins=np.arange(-50,51,4.)
    hist,xedge,yedge=np.histogram2d(added[:,0],added[:,1],bins=(bins,bins))
    ix,iy=np.unravel_index(hist.argmax(),hist.shape)
    xlo,ylo=xedge[ix],yedge[iy]
    in_patch=lambda v:(v[:,0]>=xlo)&(v[:,0]<xlo+4)&(v[:,1]>=ylo)&(v[:,1]<ylo+4)
    z=float(np.median(added[in_patch(added),2]))
    slab=lambda v:v[in_patch(v)&(abs(v[:,2]-z)<.35)]
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,title in zip(axes,('Original surface-hit labels','Closing augmentation (orange = added)')):
        points=slab(original);ax.scatter(points[:,0],points[:,1],s=18,c='#50aae6',marker='s',linewidths=0)
        if ax is axes[1]:
            points=slab(added);ax.scatter(points[:,0],points[:,1],s=18,c='#f5641e',marker='s',linewidths=0)
        ax.set(xlim=(xlo,xlo+4),ylim=(ylo,ylo+4),xlabel='X (m)',ylabel='Y (m)',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle(f'Sequence {record["sequence"]}, frame {record["frame"]:06d}; Z = {z:.2f} +/- 0.35 m')
    fig.savefig(a.output/'closing_comparison.png',dpi=160);plt.close(fig)
    atomic_json(a.output/'manifest.json',{'scope':'training GT augmentation diagnostic; not model prediction',
        'sequence':record['sequence'],'frame':record['frame'],'index':a.index,
        'geometry_sha256':g.sha256,'data_identity':identity,'transform':spec.to_dict(),**stats})


if __name__=='__main__':main()
