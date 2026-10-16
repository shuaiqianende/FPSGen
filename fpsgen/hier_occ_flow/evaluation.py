"""Full-support FN accounting, native masks, macro/micro and spatial seams."""
import numpy as np
from .voxelization import lookup_masks,unpack_children,occupied_fine_coords,pack_children
from .candidate_selection import candidate_metrics


def confusion(predicted,target):
    tp=int((predicted&target).sum()); fp=int((predicted&~target).sum()); fn=int((~predicted&target).sum())
    return {'tp':tp,'fp':fp,'fn':fn}


def rates(counts):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return {**counts,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
            'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1)}


def support_rates(predicted_ids,target_ids):
    tp=len(np.intersect1d(predicted_ids,target_ids,assume_unique=True))
    return rates({'tp':tp,'fp':len(predicted_ids)-tp,'fn':len(target_ids)-tp})


def physical_support_ids(xyz,resolution=.15):
    c=np.floor(xyz/resolution).astype(np.int64)
    return np.unique((c[:,0]+400)+1000*((c[:,1]+400)+1000*(c[:,2]+100)))


def _seams(fine_ids,geometry):
    coords=geometry.unflatten(fine_ids,fine=True); edges=[]
    for axis in range(3):
        boundary=(coords[:,axis]%geometry.child_split[axis])==geometry.child_split[axis]-1
        starts=coords[boundary]; ends=starts.copy(); ends[:,axis]+=1
        valid=(ends[:,axis]<geometry.fine_resolution[axis]); starts,ends=starts[valid],ends[valid]
        if not len(starts): continue
        b_ids=geometry.flatten(ends,fine=True)
        hit=np.isin(b_ids,fine_ids,assume_unique=True)
        edges.append(geometry.flatten(starts[hit],fine=True)*3+axis)
    return np.concatenate(edges) if edges else np.empty(0,np.int64)


def connected_surface_fragments(fine_ids,geometry):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not len(fine_ids): return {'components':0,'largest_fraction':0.,'isolated_fraction':0.}
    coords=geometry.unflatten(fine_ids,fine=True); a=[]; b=[]
    for axis in range(3):
        start=np.flatnonzero(coords[:,axis]<geometry.fine_resolution[axis]-1)
        other=coords[start].copy(); other[:,axis]+=1; ids=geometry.flatten(other,fine=True)
        where=np.searchsorted(fine_ids,ids)
        hit=(where<len(fine_ids)) & (fine_ids[np.minimum(where,len(fine_ids)-1)]==ids)
        a.append(start[hit]); b.append(where[hit])
    a,b=np.concatenate(a),np.concatenate(b)
    graph=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(fine_ids),len(fine_ids))).tocsr()
    count,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels)
    return {'components':int(count),'largest_fraction':float(sizes.max()/len(fine_ids)),
            'isolated_fraction':float((sizes==1).sum()/len(fine_ids)),'connectivity':6}


def fine_metrics(candidate_ids,predicted,label_ids,label_masks,geometry,*,observed=None,
                 observed_parent_ids=None,observed_masks=None,common_raw_gt_ids=None,structural=True):
    candidate_ids=np.asarray(candidate_ids,dtype=np.int64)
    valid=geometry.valid_children(candidate_ids)
    predicted=np.asarray(predicted,dtype=bool)&valid
    targets=unpack_children(lookup_masks(label_ids,label_masks,candidate_ids),geometry)&valid
    counts=confusion(predicted,targets)
    missed=~np.isin(label_ids,candidate_ids)
    missed_fine=int(unpack_children(label_masks[missed],geometry).sum())
    counts['fn']+=missed_fine
    empty=~targets.any(1)
    out={'fine':rates(counts),'candidate':candidate_metrics(candidate_ids,label_ids,label_masks,geometry),
      'exact_child_pattern_accuracy':float(((predicted==targets)|~valid).all(1).mean()) if len(valid) else 0.,
      'occupied_child_count_mae':float(np.abs(predicted.sum(1)-targets.sum(1)).mean()) if len(valid) else 0.,
      'parent_empty_suppression_accuracy':float((~predicted[empty].any(1)).mean()) if empty.any() else None,
      'missed_parent_fine_false_negatives':missed_fine}
    if observed is not None and observed_parent_ids is None:
        observed=np.asarray(observed,dtype=bool)&valid
        out['observed_hit']=rates(confusion(predicted[observed],targets[observed]))
        unknown=confusion(predicted[valid&~observed],targets[valid&~observed]); unknown['fn']+=missed_fine
        out['not_directly_hit']=rates(unknown)
        out['observed_policy']='Input surface hits only; unknown does not mean free space'
        out['observed_region_scope']='candidate-only observed mask; missed-parent observed status unavailable'
    pcoord=occupied_fine_coords(candidate_ids,pack_children(predicted,geometry),geometry)
    gcoord=occupied_fine_coords(label_ids,label_masks,geometry)
    pids=np.unique(geometry.flatten(pcoord,fine=True)); gids=np.unique(geometry.flatten(gcoord,fine=True))
    if observed_parent_ids is not None:
        ocoord=occupied_fine_coords(observed_parent_ids,observed_masks,geometry)
        oids=np.unique(geometry.flatten(ocoord,fine=True))
        out['observed_hit']=support_rates(np.intersect1d(pids,oids,assume_unique=True),np.intersect1d(gids,oids,assume_unique=True))
        out['not_directly_hit']=support_rates(np.setdiff1d(pids,oids,assume_unique=True),np.setdiff1d(gids,oids,assume_unique=True))
        out['observed_region_scope']='full GT support including missed-parent false negatives; input-only hit occupancy'
    ranges={}
    for a,b in ((0,10),(10,20),(20,35),(35,50)):
        def subset(coords):
            radius=np.minimum(np.linalg.norm(geometry.centers(coords,fine=True),axis=1),np.nextafter(50.,0.))
            return geometry.flatten(coords[(radius>=a)&(radius<b)],fine=True)
        pr,gr=subset(pcoord),subset(gcoord); tp=len(np.intersect1d(pr,gr,assume_unique=True))
        ranges[f'{a}-{b}']=rates({'tp':tp,'fp':len(pr)-tp,'fn':len(gr)-tp})
    out['distance_bins']=ranges
    # Cross-resolution score represents each occupied native voxel by its
    # center on the same world-origin 15cm grid. Keep native results as well.
    pr=physical_support_ids(geometry.centers(pcoord,fine=True))
    if common_raw_gt_ids is None:
        gr=physical_support_ids(geometry.centers(gcoord,fine=True))
        out['common_015m_native_center_proxy']=support_rates(pr,gr)
        out['common_grid_policy']='Native GT center proxy only; not a shared raw target across resolutions'
    else:
        out['common_015m_raw_map']=support_rates(pr,common_raw_gt_ids)
        out['common_grid_policy']='Same physical 15cm/world-origin raw map support across all native geometries'
    if structural:
        pe,ge=_seams(pids,geometry),_seams(gids,geometry); tp=len(np.intersect1d(pe,ge))
        out['parent_boundary_contact_edges']=rates({'tp':tp,'fp':len(pe)-tp,'fn':len(ge)-tp})
        out['boundary_policy']='6-neighbor occupied contact edges crossing native parent faces, compared against GT; not all-empty agreement'
        out['surface_fragments']={'prediction':connected_surface_fragments(pids,geometry),'target':connected_surface_fragments(gids,geometry)}
    return out


def aggregate_frames(rows):
    if not rows: raise ValueError('No evaluation frames')
    total={k:sum(r['fine'][k] for r in rows) for k in ('tp','fp','fn')}
    return {'frames':len(rows),'micro':rates(total),
            'macro':{key:float(np.mean([r['fine'][key] for r in rows])) for key in ('precision','recall','f1','iou')}}
