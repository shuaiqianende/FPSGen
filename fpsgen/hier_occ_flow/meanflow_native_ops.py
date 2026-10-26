"""Forward-AD bridges for the unchanged PTv3 native primal operators.

These Functions are used only inside the detached MeanFlow target JVP. Ordinary
reverse-mode training continues through the original SpConv/Flash/scatter ops.
No finite differences, frozen dynamic features, or vendor source edits.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
import torch

_ACTIVE = ContextVar('meanflow_native_jvp', default=False)


@contextmanager
def native_jvp_mode():
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def native_jvp_active():
    return _ACTIVE.get()


def sequence_groups(cumulative):
    """Prepare integer layout outside AD, preserving every sequence boundary."""
    values = tuple(int(v) for v in cumulative)
    if len(values) < 2 or values[0] != 0:
        raise ValueError('Nonempty cumulative sequence lengths starting at zero required')
    groups = []
    for start, stop in zip(values, values[1:]):
        length = stop - start
        if length <= 0:
            raise ValueError('Empty attention sequence unsupported by this bridge')
        if groups and groups[-1][1] == start and groups[-1][2] == length:
            groups[-1] = (groups[-1][0], stop, length)
        else:
            groups.append((start, stop, length))
    return tuple(groups)


def attention_tangent(qkv, dqkv, groups, scale, chunk_sequences=128):
    """Analytic d(softmax(scale QK^T)V), with the same unmasked local windows.

    The native FlashAttention call still computes the primal. The derivative
    uses FP32 statistics (FP64 for reference tests), with a bounded workspace.
    Dropout, causal/window masks and RPE are deliberately rejected at the API.
    """
    if qkv.shape != dqkv.shape or qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError('Matching [tokens,3,heads,head_dim] tensors required')
    if not groups or groups[-1][1] != len(qkv):
        raise ValueError('Prepared layout does not cover all QKV tokens')
    dtype = torch.float64 if qkv.dtype == torch.float64 else torch.float32
    parts = []
    with torch.autocast(qkv.device.type, enabled=False):
        for start, stop, length in groups:
            for first in range(start, stop, length * chunk_sequences):
                last = min(stop, first + length * chunk_sequences)
                x = qkv[first:last].reshape(-1, length, 3, *qkv.shape[2:]).to(dtype)
                dx = dqkv[first:last].reshape_as(x).to(dtype)
                q, k, v = x.permute(2, 0, 3, 1, 4).unbind(0)
                dq, dk, dv = dx.permute(2, 0, 3, 1, 4).unbind(0)
                prob = ((q @ k.transpose(-1, -2)) * scale).softmax(-1)
                ds = (dq @ k.transpose(-1, -2) + q @ dk.transpose(-1, -2)) * scale
                dp = prob * (ds - (prob * ds).sum(-1, keepdim=True))
                dy = dp @ v + prob @ dv
                parts.append(dy.transpose(1, 2).reshape(-1, *qkv.shape[2:]).to(qkv.dtype))
    return torch.cat(parts, dim=0)


class _FlashJVP(torch.autograd.Function):
    @staticmethod
    def forward(qkv, cu_seqlens, max_seqlen, scale, groups):
        from flash_attn import flash_attn_varlen_qkvpacked_func
        return flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens,
            max_seqlen=max_seqlen, dropout_p=0., softmax_scale=scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        qkv, _, _, scale, groups = inputs
        ctx.save_for_forward(qkv)
        ctx.scale, ctx.groups = scale, groups

    @staticmethod
    def jvp(ctx, dqkv, _dcu, _dmax, _dscale, _dgroups):
        qkv, = ctx.saved_tensors
        return attention_tangent(qkv, dqkv, ctx.groups, ctx.scale)

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError('Flash JVP bridge is target-only; use the original operator for reverse AD')


def flash_attention_jvp(qkv, cu_seqlens, max_seqlen, scale, groups):
    return _FlashJVP.apply(qkv, cu_seqlens, max_seqlen, scale, groups)


class _SparseCall:
    # An opaque metadata object, not a pytree dict reconstructed by functorch.
    def __init__(self, module, source, bias=True):
        self.module, self.source, self.bias, self.result = module, source, bias, None


def _sparse_eval(features, packet):
    module = packet.module
    source = packet.source.replace_feature(features)
    if packet.bias:
        packet.result = module._mf_native_forward(source)
    else:
        packet.result = module._conv_forward(module.training, source, module.weight,
            None, name=module.name, sparse_unique_name=module._sparse_unique_name,
            act_type=module.act_type, act_alpha=module.act_alpha, act_beta=module.act_beta)
    return packet.result.features


class _SparseTangentEval(torch.autograd.Function):
    @staticmethod
    def forward(features, packet):
        return _sparse_eval(features, packet)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError('Sparse tangent evaluation is detached target-only')


class _SubMConvJVP(torch.autograd.Function):
    @staticmethod
    def forward(features, packet):
        return _sparse_eval(features, packet)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.packet = inputs[1]

    @staticmethod
    def jvp(ctx, dfeatures, _dpacket):
        packet = ctx.packet
        module = packet.module
        source = packet.source.replace_feature(dfeatures)
        # Same immutable coordinate/index tensors, a separate mutable dict.
        source.indice_dict = packet.result.indice_dict.copy()
        return _SparseTangentEval.apply(dfeatures, _SparseCall(module, source, bias=False))

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError('SubM JVP bridge is target-only; use the original operator for reverse AD')


def subm_conv_jvp(module, source):
    if not module.subm or module.inverse or module.transposed:
        raise ValueError('Only the actual PTv3 SubM convolution is registered')
    if getattr(module.act_type, 'value', int(module.act_type)) != 0:
        raise ValueError('Fused sparse activation requires its own derivative')
    packet = _SparseCall(module, source)
    features = _SubMConvJVP.apply(source.features, packet)
    return packet.result.replace_feature(features)


class _SegmentMaxJVP(torch.autograd.Function):
    @staticmethod
    def forward(features, ptr):
        import torch_scatter
        return torch_scatter.segment_max_csr(features, ptr)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, arg = output
        ctx.save_for_forward(arg)
        ctx.mark_non_differentiable(arg)
        ctx.rows = len(inputs[0])

    @staticmethod
    def jvp(ctx, dfeatures, _dptr):
        arg, = ctx.saved_tensors
        # The native arg indices define the same tie subgradient as reverse AD.
        padded = torch.cat((dfeatures, torch.zeros_like(dfeatures[:1])), dim=0)
        return padded.gather(0, arg.clamp(max=ctx.rows)), None

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError('Segment max bridge is target-only; use original scatter for reverse AD')


def segment_max_jvp(features, ptr):
    return _SegmentMaxJVP.apply(features, ptr)[0]


def _sparse_forward(module, source, add_input=None):
    if not native_jvp_active():
        return module._mf_native_forward(source, add_input)
    if add_input is not None:
        raise ValueError('Fused add_input is not present in the registered PTv3')
    return subm_conv_jvp(module, source)


def _attention_forward(module, point):
    if not native_jvp_active():
        if getattr(module, '_mf_capture_native_layout', False):
            _, _, cu = module.get_padding_and_inverse(point)
            module._mf_flash_groups = sequence_groups(cu.detach().cpu().tolist())
        return module._mf_native_forward(point)
    if not module.enable_flash or module.enable_rpe or module.attn_drop != 0:
        raise ValueError('Only the registered no-dropout/no-RPE FlashAttention path is supported')
    groups = module._mf_flash_groups
    if groups is None:
        raise ValueError('Flash sequence layout must be prepared outside JVP')
    pad, unpad, cu = module.get_padding_and_inverse(point)
    order = point.serialized_order[module.order_index][pad]
    inverse = unpad[point.serialized_inverse[module.order_index]]
    qkv = module.qkv(point.feat)[order]
    y = flash_attention_jvp(qkv.half().reshape(-1, 3, module.num_heads,
        module.channels // module.num_heads), cu, module.patch_size, module.scale, groups)
    y = y.reshape(-1, module.channels).to(qkv.dtype)[inverse]
    point.feat = module.proj_drop(module.proj(y))
    return point


def install_native_jvp_bridges(net):
    import spconv.pytorch as spconv
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import SerializedAttention
    attention = {}
    for name, module in net.named_modules():
        if isinstance(module, spconv.SubMConv3d):
            if not hasattr(module, '_mf_native_forward'):
                module._mf_native_forward = module.forward
                module.forward = MethodType(_sparse_forward, module)
        elif isinstance(module, SerializedAttention):
            if not hasattr(module, '_mf_native_forward'):
                module._mf_native_forward = module.forward
                module._mf_capture_native_layout = False
                module._mf_flash_groups = None
                module.forward = MethodType(_attention_forward, module)
            attention[name] = module
    return attention
