"""Opt-in, point-aligned state/condition residuals outside PTv3 checkpoint blocks.

No vendor change or global patch is needed. Injection runs before the existing
checkpointed block; backward therefore sees the ordinary residual graph once,
while the block's original CPE/attention/AdaLN/MLP are recomputed unchanged.
"""
import torch
from torch import nn


class _PointInjection:
    # A callable object, not a closure: deepcopy/EMA must resolve the reference
    # to the copied interaction module instead of reading online parameters.
    def __init__(self, interaction, index):
        self.interaction = interaction
        self.index = index

    def __call__(self, block, args):
        point = args[0]
        residual = point.get('state_condition_residual')
        if residual is None or residual.shape != point.feat.shape:
            raise ValueError('State-condition residual must match original-resolution tokens')
        projection = self.interaction.projections[self.index]
        # The residual is built in the outer FP16 autocast region, whereas
        # the original PTv3 trunk explicitly disables autocast for FP32 ops.
        branch = projection(residual.to(projection.weight.dtype))
        branch = branch.to(point.feat.dtype)
        gate = self.interaction.scales[self.index].to(point.feat.dtype)
        point.feat = point.feat + gate * branch
        if 'sparse_conv_feat' in point:
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return (point,)


class StateConditionInteraction(nn.Module):
    block_names = ('enc.enc0.block0', 'dec.dec0.block1')

    def __init__(self, static_channels=144, dynamic_channels=96, width=64):
        super().__init__()
        self.static_encoder = nn.Sequential(nn.Linear(static_channels, width), nn.SiLU(), nn.Linear(width, width))
        self.dynamic_encoder = nn.Sequential(nn.Linear(dynamic_channels, width), nn.SiLU(), nn.Linear(width, width))
        self.condition_gate = nn.Linear(width, width)
        self.residual = nn.Linear(width, width)
        self.projections = nn.ModuleList(nn.Linear(width, width) for _ in self.block_names)
        self.scales = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in self.block_names)
        self._installed = False

    def forward(self, static_features, dynamic_features):
        if static_features.shape[0] != dynamic_features.shape[0]:
            raise ValueError('State and condition must share the same Parent ordering')
        condition = self.static_encoder(static_features)
        dynamic = self.dynamic_encoder(dynamic_features)
        return self.residual(dynamic * torch.sigmoid(self.condition_gate(condition)))

    def install(self, net):
        if self._installed:
            raise ValueError('Interaction hooks are installed exactly once per model')
        blocks = dict(net.named_modules())
        for index, name in enumerate(self.block_names):
            if name not in blocks:
                raise ValueError(f'Missing original-resolution PTv3 block: {name}')
            block = blocks[name]
            if getattr(block, 'channels', None) != self.residual.out_features:
                raise ValueError(f'Interaction block width mismatch: {name}')
            block.register_forward_pre_hook(_PointInjection(self, index))
        self._installed = True


def load_shared_baseline(model, state):
    """Load every baseline tensor exactly; allow only new M1 keys to be absent."""
    expected = model.state_dict()
    extra = set(state) - set(expected)
    missing = set(expected) - set(state)
    allowed=('interaction.',)
    if hasattr(model,'child_state_encoder'):allowed+=('child_state_encoder.',)
    if extra or any(not key.startswith(allowed) for key in missing):
        raise ValueError(f'Baseline/model keys differ: extra={sorted(extra)}, missing={sorted(missing)}')
    for key, value in state.items():
        if expected[key].shape != value.shape:
            raise ValueError(f'Baseline/model shape differs at {key}')
    model.load_state_dict(state, strict=not missing)
    # Caller must not feed a previously trained branch as a zero-parity model.
    if hasattr(model, 'interaction') and any(float(x.detach()) != 0 for x in model.interaction.scales):
        raise ValueError('New M1 residual scales must start at zero')
    if hasattr(model,'child_state_encoder') and bool((model.child_state_encoder.output.weight.detach()!=0).any()):
        raise ValueError('New local state output projection must start at zero')
    return {'shared_tensors': len(state), 'new_keys': sorted(missing)}
