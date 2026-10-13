"""Hurdle adaptation: occupancy probability times conditional positive count."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .voxdet_encoder import VoxDetEncoder
from fpsgen.voxel_point.density_loss import density_loss


class BaseDensity(nn.Module):
    def __init__(self,geometry,*,activation_checkpointing=True):
        super().__init__()
        self.geometry=geometry
        self.encoder=VoxDetEncoder(geometry,activation_checkpointing=activation_checkpointing)
        # Independent occupancy head on the same64-channel refined features.
        # It can express high conditional count with low occupancy confidence.
        self.occupancy_head=nn.Conv3d(64,1,1)

    def initialize_density_checkpoint(self,path):
        value=torch.load(path,map_location='cpu')
        if value['stage']!='stage1':
            raise ValueError("Warmstart must be a Stage1 density checkpoint")
        for key,v in value['geometry'].items():
            if self.geometry.to_dict()[key]!=v:
                raise ValueError("Warmstart spatial grid mismatch")
        self.encoder.load_state_dict(value['model'],strict=True)
        count_head=self.encoder.main_head.cls_convs[0][-1]
        with torch.no_grad():
            self.occupancy_head.weight.copy_(2*count_head.weight)
            self.occupancy_head.bias.copy_(2*count_head.bias-2*math.log(math.expm1(1.)))
        return {'upstream_checkpoint_updates':value['successful_updates'],
                'new_occupancy_head_initialized':True,'new_gt_sphere_protocol':True}

    def forward(self,input_xyz):
        voxel,cls,geometry,conditions=self.encoder.encode_features(input_xyz)
        features=[]
        # Scoped observation of an unchanged official module; the hook performs
        # no operations and is removed before checkpoint recomputation.
        hook=self.encoder.main_head.cls_convs[0][-2].register_forward_hook(lambda module,args,result:features.append(result))
        try:
            output=self.encoder._run(self.encoder.main_head,[[cls[0]],[geometry[0]]])
        finally:
            hook.remove()
        if len(features)!=1:raise ValueError('Expected one official refined classification feature')
        raw=output['output_voxels'].float()
        logits=self.occupancy_head(features[0])
        logits=F.interpolate(logits,size=self.geometry.shape,mode='trilinear',align_corners=False).float()
        probability=logits.sigmoid()
        positive=F.softplus(raw)
        valid=self.geometry.valid_circle_mask(device=raw.device)[None,None]
        result={'density':self.geometry.to_field(probability*positive*valid),
                'occupancy_probability':self.geometry.to_field(probability*valid),
                'occupancy_logits':self.geometry.to_field(logits),
                'positive_count':self.geometry.to_field(positive),
                'condition':tuple(self.geometry.condition_to_field(c) for c in conditions)}
        if self.training:
            aux=self.encoder._run(self.encoder.aux_head,[voxel])
            result['offset_xyz']=output['output_bbox']
            result['aux_density_xyz']=F.softplus(aux['output_voxels'].float())*valid
        return result


def base_loss(output,counts,geometry,*,occupancy_weight=1.,positive_count_weight=.3,density_weights=None):
    valid=geometry.field_mask(device=counts.device)
    target=(counts>0)&valid
    bce=F.binary_cross_entropy_with_logits(output['occupancy_logits'].float(),target.float(),reduction='none')
    pos=(bce*target).sum()/target.sum().clamp_min(1)
    bg=(bce*((~target)&valid)).sum()/((~target)&valid).sum().clamp_min(1)
    occupancy=.5*(pos+bg)
    error=F.smooth_l1_loss(output['positive_count'].float().log1p(),counts.float().log1p(),reduction='none')
    count=(error*target).sum()/target.sum().clamp_min(1)
    adapted={'pred_density':geometry.to_xyz(output['density']),
             'offset':output['offset_xyz'],'aux_density':output['aux_density_xyz']}
    density=density_loss(adapted,geometry.to_xyz(counts),geometry,weights=density_weights)
    return {'total':density['total']+occupancy_weight*occupancy+positive_count_weight*count,
            'occupancy':occupancy,'conditional_positive_count':count,'density':density['total']}
