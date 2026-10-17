"""384-only canonical variable-length scenes, unchanged network capacity.

Legacy HierarchicalOccupancyModel and vendor PointFlow normalization stay intact.
Canonicalizing BEFORE the input MLPs also stabilizes mixed-precision GEMM row
packing. A scene mean/variance reduction replaces atomic index_add for the
single-scene-per-rank instance normalization used by this experiment.
"""
import torch
from .model import HierarchicalOccupancyModel


def single_scene_norm(base):
    class SceneNorm(base):
        def forward(self,point):
            features=point.feat
            if len(features)==0:return point
            # Pooling/unpooling preserves contiguous scene groups. Reduce each
            # scene independently without atomic scatter accumulation or padding.
            groups,counts=torch.unique_consecutive(point.batch,return_counts=True)
            if len(torch.unique(groups))!=len(groups):raise ValueError('Scene normalization requires contiguous scene groups')
            values=[]
            for scene in features.split(counts.tolist()):
                if len(scene)==1:values.append(scene)
                else:
                    centered=scene-scene.mean(0,keepdim=True)
                    values.append(centered*torch.rsqrt(centered.square().mean(0,keepdim=True)+self.eps))
            normalized=torch.cat(values)
            if self.affine:normalized=normalized*self.weight+self.bias
            point.feat=normalized
            if 'sparse_conv_feat' in point:point.sparse_conv_feat=point.sparse_conv_feat.replace_feature(normalized)
            return point
    return SceneNorm


class Stage2OccupancyModel(HierarchicalOccupancyModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.backbone_kind=='ptv3':
            from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointInstanceNorm,SerializedPooling
            norm_type=single_scene_norm(PointInstanceNorm)
            for module in self.trunk.modules():
                if isinstance(module,PointInstanceNorm):module.__class__=norm_type
                # Official top-level shuffle_orders=False is NOT forwarded to
                # SerializedPooling, whose default stays True even in eval.
                if isinstance(module,SerializedPooling):module.shuffle_orders=False

    def forward(self,parent_ids,*,batch=None,state=None,time=None,lidar_features=None,parent_score=None,observed_child=None,valid_child_mask=None):
        if batch is None:batch=torch.zeros_like(parent_ids)
        if batch.shape!=parent_ids.shape or not len(parent_ids):raise ValueError('Nonempty packed parent IDs and scene batch required')
        scenes=int(batch.max())+1
        if not torch.equal(torch.unique(batch),torch.arange(scenes,device=batch.device)):raise ValueError('Contiguous scene IDs required')
        keys=batch*self.geometry.num_parents+parent_ids;order=torch.argsort(keys);inverse=torch.argsort(order)
        if len(keys)>1 and bool((keys[order][1:]==keys[order][:-1]).any()):raise ValueError('Duplicate parent tokens within a scene')
        parent_ids=parent_ids[order];batch=batch[order]
        coords=self.geometry.unflatten(parent_ids);centers=self.geometry.centers(coords).float()
        position=2*(centers-centers.new_tensor(self.geometry.origin))/centers.new_tensor([b-a for a,b in zip(self.geometry.origin,self.geometry.maximum)])-1
        if parent_score is None or valid_child_mask is None:raise ValueError('Explicit parent score and valid mask required')
        parts=[self.position(position),self.score(parent_score[order].float().reshape(-1,1))]
        if hasattr(self,'lidar'):
            if lidar_features is None:raise ValueError('Actual LiDAR pyramid required')
            parts.append(self.lidar(lidar_features[order].float()))
        if hasattr(self,'observed'):
            if observed_child is None:raise ValueError('Input-only observed mask required')
            parts.append(self.observed(observed_child[order].float()))
        embedding=None
        if self.objective=='fm':
            if state is None or state.shape!=(len(parent_ids),self.geometry.k):raise ValueError('FM state must be [sum M,K]')
            if time is None or time.numel()!=scenes or not bool(((time>=0)&(time<=1)).all()):raise ValueError('One flow time per scene required')
            state=state[order];angle=time.float().reshape(scenes,1)*self.frequencies[None]
            embedding=self.time(torch.cat((angle.sin(),angle.cos()),-1))
            parts.extend((self.local(state.float()),embedding[batch]))
        elif state is not None or time is not None:raise ValueError('BCE has no source or time')
        features=self.fuse(torch.cat(parts,-1))
        if self.backbone_kind=='ptv3':
            from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point
            data={'feat':features.float(),'coord':centers,'grid_coord':coords.int(),'batch':batch.long(),
                  'sparse_shape':[n+1 for n in self.geometry.parent_resolution]}
            if embedding is not None:data['context']=embedding.float()
            # Fixed geometry depth avoids a different scene changing Hilbert
            # orientation. Batch bits keep serialization and pooling disjoint.
            net=self.trunk.net;point=Point(data)
            point.serialization(order=net.order,depth=(max(self.geometry.parent_resolution)-1).bit_length(),shuffle_orders=False)
            with torch.autocast(features.device.type,enabled=False):
                point.sparsify();point=net.embedding(point);point=net.enc(point);point=net.dec(point)
            features=point.feat
        else:features=features+self.trunk(features)
        decoded=[features]
        if self.objective=='fm':decoded.extend((state.float(),embedding[batch]))
        return (self.decoder(torch.cat(decoded,-1))*valid_child_mask[order])[inverse]
