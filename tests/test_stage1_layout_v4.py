import torch
from torch import nn

from fpsgen.hier_occ_flow.stage1_layout import enable_channels_last_3d, remove_layout_hooks


def test_channels_last_helper_only_formats_5d_conv_weights_and_inputs():
    module = nn.Sequential(nn.Conv3d(2, 4, 3, padding=1), nn.BatchNorm3d(4))
    handles = enable_channels_last_3d(module)
    try:
        conv = module[0]
        assert conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
        x = torch.randn(1, 2, 4, 5, 6)
        seen = []
        handle = conv.register_forward_pre_hook(
            lambda _layer, args: seen.append(args[0].is_contiguous(memory_format=torch.channels_last_3d))
        )
        y = module(x)
        handle.remove()
        assert y.shape == (1, 4, 4, 5, 6)
        assert seen == [True]
    finally:
        remove_layout_hooks(handles)


def test_channels_last_helper_does_not_reformat_vector_parameters():
    module = nn.Sequential(nn.Conv3d(2, 4, 1), nn.Linear(4, 3))
    before = module[1].weight.detach().clone()
    handles = enable_channels_last_3d(module)
    try:
        assert torch.equal(module[1].weight, before)
        assert module[1].weight.ndim == 2
    finally:
        remove_layout_hooks(handles)
