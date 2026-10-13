"""Preserve pinned VoxDet modules; expose coarse conditions within Stage1 only."""

import torch
from fpsgen.voxel_point.voxdet_density import VoxDetDensity, lidar_input_features
from .checkpointing import checkpoint_once_batchnorm


class VoxDetEncoder(VoxDetDensity):
    def _run(self,module,*args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint_once_batchnorm(module,*args)
        return module(*args)

    def encode_features(self,input_xyz):
        scenes=[input_xyz] if isinstance(input_xyz,torch.Tensor) else list(input_xyz)
        tokens=[]
        for xyz in scenes:
            features,indices,_=lidar_input_features(xyz,self.spec)
            tokens.append(self.lidar_tokenizer(features,indices))
        token=torch.cat(tokens,0)
        pyramid=self._run(self.voxel_backbone,token)
        voxel=self._run(self.voxel_neck,pyramid)[0]
        encoded=self._run(self.local_backbone,voxel)
        cls,geometry=self._run(self.local_neck,encoded)
        return voxel,cls,geometry,(cls[0],pyramid[1],pyramid[2])
