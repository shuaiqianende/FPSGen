"""Geometry-ordered eight-Child state residual; never creates extra PTv3 tokens."""
import torch
from torch import nn


class ChildLocalStateEncoder(nn.Module):
    def __init__(self,geometry,output_width=64,*,mode='position',width=16,heads=2):
        super().__init__()
        if geometry.child_split!=(2,2,2) or mode not in ('position','position_attention'):
            raise ValueError('Local state study requires eight XYZ-ordered Children')
        if width%heads:raise ValueError('Local width must be divisible by heads')
        self.mode=mode;self.width=width;self.heads=heads
        # Geometry uses x-fastest k=x+2*y+4*z, not a ZYX tensor reshape.
        offsets=torch.as_tensor(geometry.child_offsets(),dtype=torch.float32)
        centers=2*((offsets+.5)/offsets.new_tensor(geometry.child_split)-.5)
        self.register_buffer('relative_centers',centers)
        self.state_embedding=nn.Linear(1,width)
        self.position_embedding=nn.Linear(3,width,bias=False)
        if mode=='position_attention':
            self.attention_norm=nn.LayerNorm(width)
            self.qkv=nn.Linear(width,3*width)
            self.attention_output=nn.Linear(width,width)
        self.ffn_norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))
        # Ordered flattening preserves each Child slot. Zero output projection
        # preserves the existing source model exactly at initialization.
        self.output=nn.Linear(8*width,output_width,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,state,valid_child_mask):
        if state.ndim!=2 or state.shape[1]!=8 or valid_child_mask.shape!=state.shape:
            raise ValueError('Need aligned [M,8] flow state and valid Child mask')
        if valid_child_mask.dtype!=torch.bool:raise ValueError('Valid Child mask must be Boolean')
        weight_dtype=self.state_embedding.weight.dtype
        mask=valid_child_mask[...,None]
        # Mask before embedding as well: unused invalid-slot values cannot
        # leak into legal Child features, including when they contain NaNs.
        values=torch.where(valid_child_mask,state,torch.zeros_like(state)).to(weight_dtype)
        pos=self.position_embedding(self.relative_centers.to(weight_dtype))[None]
        tokens=(self.state_embedding(values[...,None])+pos)*mask
        if self.mode=='position_attention':
            n=len(tokens);d=self.width//self.heads
            qkv=self.qkv(self.attention_norm(tokens)).reshape(n,8,3,self.heads,d)
            q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
            logits=(q@k.transpose(-2,-1)).float()*(d**-.5)
            keys=valid_child_mask[:,None,None,:]
            probabilities=logits.masked_fill(~keys,torch.finfo(logits.dtype).min).softmax(-1)
            probabilities=probabilities*keys
            probabilities=probabilities/probabilities.sum(-1,keepdim=True).clamp_min(1e-12)
            attended=(probabilities.to(v.dtype)@v).transpose(1,2).reshape(n,8,self.width)
            tokens=(tokens+self.attention_output(attended))*mask
        tokens=(tokens+self.ffn(self.ffn_norm(tokens)))*mask
        return self.output(tokens.flatten(1))
