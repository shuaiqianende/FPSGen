"""Checkpoint complete PWM time-conditioned blocks without changing parameters."""

import torch
from torch.utils.checkpoint import checkpoint


def enable_block_checkpointing(net):
    from .vendor.pwm.ptv3_time_conditioning import TimeConditionedBlock
    from .vendor.pwm.ptv3_reference.model import Point

    class CheckpointedBlock(TimeConditionedBlock):
        def forward(self, point):
            if not self.training or not torch.is_grad_enabled():
                return super().forward(point)
            # Point is mutable. Replay into a fresh container, using identical
            # serialization/coordinates and the block's input sparse structure.
            # Transformer blocks never pool/unpool. Capturing hierarchy parents
            # here would create grad_fn → Point parent → decoder grad_fn cycles.
            metadata = {key:value for key,value in point.items() if key not in
                ('feat','context','sparse_conv_feat','pooling_parent','unpooling_parent')}
            sparse = point.sparse_conv_feat
            context = point.get('context', point.feat.new_empty((0,32)))

            def run(features, sparse_features, scene_context):
                replay = Point(metadata.copy())
                replay.feat = features
                # Official unpooling updates .feat before syncing sparse features.
                # Preserve BOTH inputs to CPE rather than silently syncing them.
                replay.sparse_conv_feat = sparse.replace_feature(sparse_features)
                if scene_context.numel():
                    replay.context = scene_context
                return super(CheckpointedBlock,self).forward(replay).feat

            # Torch 2.0's nonreentrant saved-tensor hooks trigger a FlashAttention
            # saved-leaf assertion. The tensor-only block interface supports the
            # established reentrant implementation (features require gradients).
            point.feat = checkpoint(run,point.feat,sparse.features,context,use_reentrant=True)
            point.sparse_conv_feat = sparse.replace_feature(point.feat)
            return point

    for block in net.modules():
        if isinstance(block,TimeConditionedBlock):
            block.__class__ = CheckpointedBlock
    return net
