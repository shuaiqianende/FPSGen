"""Native AD probes on actual PTv3 operators; failures are retained independently."""
import traceback
import torch
from .meanflow_jvp import without_activation_checkpointing


def probe_stage2_components(model,context):
    """No alternative attention masks, dtypes or sparse math. GPU gate only."""
    from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import Point,SerializedAttention,SerializedPooling,SerializedUnpooling
    import spconv.pytorch as spconv
    base=model.base;net=base.trunk.net;device=context.coords.device;rows=len(context.coords)
    x=torch.randn(rows,64,device=device,dtype=torch.float32);results=[]
    def probe(name,fn,value):
        try:
            with model._layouts(context,'static'),without_activation_checkpointing(model),torch.no_grad(),torch.autocast('cuda',enabled=False):
                primal,tangent=torch.func.jvp(fn,(value,),(torch.ones_like(value),))
            results.append(dict(name=name,status='JVP_PASS',finite=bool(torch.isfinite(primal).all() and torch.isfinite(tangent).all()),
                derivative_max_abs=float(tangent.abs().max()),shape=list(primal.shape)))
        except Exception as exc:
            results.append(dict(name=name,status='BLOCKED_JVP',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
    probe('local_state_MLP',lambda z:base.local(z),torch.randn(rows,8,device=device))
    probe('time_MLP',lambda z:base.time(z),torch.randn(context.scenes,32,device=device))
    first=next(m for m in net.modules() if isinstance(m,SerializedAttention))
    probe('attention_qkv_linear',lambda z:first.qkv(z),x)
    def point(features):
        p=context.topology.make_point(features,torch.zeros(context.scenes,32,device=device))
        p['_validated_batch_size']=context.scenes
        p['sparse_conv_feat']=spconv.SparseConvTensor(features,context.sparse_indices,list(context.topology.sparse_shape),context.scenes)
        return p
    probe('SpConv_embedding_and_norm',lambda z:net.embedding(point(z)).feat,x)
    probe('serialized_attention_actual_FlashAttention',lambda z:first(point(z)).feat,x)
    block=net.enc.enc0.block0
    probe('SpConv_CPE',lambda z:block.cpe(point(z)).feat,x)
    # Exercise the unchanged time-AdaLN block (including its attention/CPE),
    # rather than an invented small dense replacement.
    with model._layouts(context,'static'):
        probe('full_time_conditioned_block',lambda z:block(point(z)).feat,x)
        pooling=next(m for m in net.modules() if isinstance(m,SerializedPooling))
        probe('pooling_project_and_segment_reduce',lambda z:pooling(point(z)).feat,x)
        unpool=next(m for m in net.modules() if isinstance(m,SerializedUnpooling))
        try:
            with torch.no_grad():pooled=pooling(point(x))
            metadata={k:v for k,v in pooled.items() if k not in ('feat','sparse_conv_feat')};sparse=pooled.sparse_conv_feat
            def unpool_fn(z):
                p=Point(metadata.copy());p.feat=z;p.sparse_conv_feat=sparse.replace_feature(z)
                # unpooling pops only this new Point, never the captured parent.
                return unpool(p).feat
            probe('unpool_project_and_index_gather',unpool_fn,pooled.feat)
        except Exception as exc:results.append(dict(name='unpool_preparation',status='BLOCKED_PRIMAL',reason=str(exc)))
    return results
