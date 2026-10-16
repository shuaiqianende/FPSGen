"""One official PTv3 token per parent; absolute parent grid, no XYZ velocity head."""
import torch
from torch import nn


def checkpoint_parent_blocks(net):
    from torch.utils.checkpoint import checkpoint
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Block,Point
    variants={}
    def make_variant(base):
        class Checkpointed(base):
            def forward(self,point):
                if not self.training or not torch.is_grad_enabled(): return super().forward(point)
                metadata={k:v for k,v in point.items() if k not in ('feat','context','sparse_conv_feat','pooling_parent','unpooling_parent')}
                sparse=point.sparse_conv_feat; context=point.get('context',point.feat.new_empty((0,32)))
                def run(features,sparse_features,time):
                    replay=Point(metadata.copy()); replay.feat=features
                    replay.sparse_conv_feat=sparse.replace_feature(sparse_features)
                    if time.numel(): replay.context=time
                    return super(Checkpointed,self).forward(replay).feat
                point.feat=checkpoint(run,point.feat,sparse.features,context,use_reentrant=True)
                point.sparse_conv_feat=sparse.replace_feature(point.feat)
                return point
        return Checkpointed
    for block in list(net.modules()):
        if isinstance(block,Block):
            base=type(block)
            if base not in variants: variants[base]=make_variant(base)
            block.__class__=variants[base]
    return net


class ParentPTv3(nn.Module):
    def __init__(self,width=64,*,checkpointing=True,patch_sizes=(64,128),depths=(2,2),enable_flash=True,time_conditioned=True):
        super().__init__()
        from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointTransformerV3
        from fpsgen.voxel_point.vendor.pwm.ptv3_time_conditioning import build_official_ptv3_with_time_context
        def build(): return PointTransformerV3(
            in_channels=width,order=('z','z-trans','hilbert','hilbert-trans'),stride=(2,),
            enc_depths=tuple(depths),enc_channels=(width,2*width),enc_num_head=(4,8),enc_patch_size=tuple(patch_sizes),
            dec_depths=(2,),dec_channels=(width,),dec_num_head=(4,),dec_patch_size=(patch_sizes[0],),
            shuffle_orders=False,drop_path=0.,enable_flash=enable_flash,point_norm='instance',
            upcast_attention=False,upcast_softmax=False)
        self.net=build_official_ptv3_with_time_context(build) if time_conditioned else build()
        if checkpointing:
            checkpoint_parent_blocks(self.net)

    def forward(self,features,parent_coords,world_centers,scene_time,parent_resolution):
        if len(parent_coords)!=len(features) or len(features)==0: raise ValueError('Nonempty parent tokens required')
        ids=parent_coords[:,0]+parent_resolution[0]*(parent_coords[:,1]+parent_resolution[1]*parent_coords[:,2])
        order=torch.argsort(ids)
        if len(ids)>1 and bool((ids[order][1:]==ids[order][:-1]).any()): raise ValueError('Duplicate parent tokens')
        # Sorting stabilizes tie-breaking/order-independent inference. Explicit
        # absolute grid_coord bypasses official coord.min(0) re-quantization.
        data={'feat':features[order].float(),'coord':world_centers[order].float(),
              'grid_coord':parent_coords[order].int(),'batch':torch.zeros(len(ids),device=ids.device,dtype=torch.long),
              'sparse_shape':[n+1 for n in parent_resolution]}
        if scene_time is not None: data['context']=scene_time.reshape(1,32).float()
        # Same proven SpConv eval precision policy as existing PointFlow.
        with torch.autocast(features.device.type,enabled=False): decoded=self.net(data).feat
        inverse=torch.argsort(order)
        return decoded[inverse]
