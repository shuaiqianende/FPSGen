from torch import nn
import pytest
import torch
from fpsgen.hier_occ_flow.dense_encoder_variants import ConvNormAct3d,forward_variant_backbone
from fpsgen.hier_occ_flow.encoder_adaptation import parameter_sha
from fpsgen.hier_occ_flow.encoder_retraining_preparation import prepare_selective_stage0


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=ConvNormAct3d(128,128,3,padding=1,activation=True)
        self.conv2=ConvNormAct3d(128,128,3,padding=1);self.conv2.norm_name='norm'
        self.downsample=None;self.drop_path=nn.Identity();self.relu=nn.ReLU()
    def forward(self,x):return self.relu(self.conv2(self.conv1(x))+x)


class Backbone(nn.Module):
    def __init__(self,count):
        super().__init__();self.layers=nn.Sequential(nn.Sequential(*[Block() for _ in range(count)]),nn.Identity())
        self.backbone_output_ids=(0,1)
    def forward(self,x):
        first=self.layers[0](x);return [first,self.layers[1](first)]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.voxel_backbone=Backbone(2);self.local_backbone=Backbone(3);self.encoder_variant='original'


@pytest.mark.parametrize('target',['ER-Local','ER-Voxel'])
def test_selective_adapter_changes_only_registered_stage0_and_preserves_b2_gradients(target):
    torch.manual_seed(6);encoder=Encoder()
    untouched=encoder.voxel_backbone if target=='ER-Local' else encoder.local_backbone
    untouched_sha=parameter_sha(untouched);audit=prepare_selective_stage0(encoder,target)
    assert parameter_sha(untouched)==untouched_sha and not audit['formal_training_go']
    chosen=encoder.local_backbone if target=='ER-Local' else encoder.voxel_backbone
    x=torch.randn(2,128,3,3,2,requires_grad=True);outputs=forward_variant_backbone(chosen,x,None)
    assert all(y.shape==x.shape for y in outputs)
    outputs[-1].square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    with pytest.raises(ValueError,match='original scratch'):prepare_selective_stage0(encoder,target)
