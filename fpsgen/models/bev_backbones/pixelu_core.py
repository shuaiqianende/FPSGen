"""PixelU-B topology, width-scaled for FPSGen BEV Flow.

Architecture follows ``gzp6688/PixelU`` at
``3b7733d931cbce72cf5f4c5e1ccb0c523d9b65c6``, source ``model_pixelu.py``.
``PixelU-S-BEV`` keeps the published B-16 topology (patch-16, [4,4,4]
stages, one PixelUnshuffle/PixelShuffle pair and skip-all-blocks) while only
scaling width from 768 to 384.  ``torch.compile`` decorators are deliberately
not applied here; compilation is a later, separately controlled experiment.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """RMSNorm used by PixelU's JiT blocks and QK normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * inv.to(x.dtype)) * self.weight.to(x.dtype)


class VisionRotaryEmbeddingFast(nn.Module):
    """2-D RoPE for PixelU image tokens, preserving optional context prefixes."""

    def __init__(self, dim: int, pt_seq_len: int, num_cls_token: int = 0,
                 theta: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE half-head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        coords = torch.arange(pt_seq_len).float()
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        phase = torch.cat((torch.outer(y.flatten(), inv_freq),
                           torch.outer(x.flatten(), inv_freq)), dim=-1)
        phase = torch.repeat_interleave(phase, 2, dim=-1)
        self.num_cls_token = int(num_cls_token)
        self.register_buffer("cos", phase.cos()[None, None], persistent=False)
        self.register_buffer("sin", phase.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix = x[:, :, :self.num_cls_token] if self.num_cls_token else None
        image = x[:, :, self.num_cls_token:]
        if image.shape[-2] != self.cos.shape[-2]:
            raise ValueError(f"RoPE expected {self.cos.shape[-2]} image tokens, got {image.shape[-2]}")
        cos, sin = self.cos.to(image.dtype), self.sin.to(image.dtype)
        image = image * cos + self._rotate_half(image) * sin
        return torch.cat((prefix, image), dim=2) if prefix is not None else image


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine token positions, matching PixelU's frozen map."""
    if embed_dim % 4:
        raise ValueError("2-D sine-cosine embedding dimension must divide by four")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))
    grid = torch.arange(grid_size, dtype=torch.float32)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)), dim=1)


class Downsample(nn.Module):
    """Official token-to-image Conv2d + PixelUnshuffle downsample."""

    def __init__(self, n_feat: int, out_feat: int | None = None):
        super().__init__()
        out_feat = n_feat if out_feat is None else out_feat
        if out_feat % 4:
            raise ValueError("PixelUnshuffle output channels must be divisible by four")
        self.out_feat = int(out_feat)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_feat // 4, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.body(x).flatten(2).transpose(1, 2)


class Upsample(nn.Module):
    """Official token-to-image Conv2d + PixelShuffle upsample."""

    def __init__(self, n_feat: int, out_feat: int):
        super().__init__()
        self.out_feat = int(out_feat)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_feat * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.body(x).flatten(2).transpose(1, 2)


class BottleneckPatchEmbed(nn.Module):
    """Official patch-stride bottleneck projection followed by a 1x1 lift."""

    def __init__(self, input_size: int, patch_size: int, in_channels: int,
                 bottleneck_dim: int, hidden_size: int):
        super().__init__()
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj1 = nn.Conv2d(in_channels, bottleneck_dim, patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size}, got {tuple(x.shape[-2:])}")
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(
            half, dtype=torch.float32, device=t.device
        ) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1) if dim % 2 else emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Attention(nn.Module):
    """PixelU QK-RMSNorm RoPE scaled-dot-product attention."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 qk_norm: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden size must divide number of attention heads")
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        x = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(first) * second))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class JiTBlock(nn.Module):
    """Official JiT AdaLN, RoPE attention and SwiGLU block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.,
                 attn_drop: float = 0., proj_drop: float = 0., skip: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        self.skip_linear = nn.Linear(hidden_size * 2, hidden_size) if skip else None

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat((x, skip), dim=-1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa[:, None] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
        return x + gate_mlp[:, None] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class UiTCore(nn.Module):
    """PixelU-B-16 U-shaped token topology, without REPA or class labels."""

    def __init__(self, input_size: int = 256, patch_size: int = 16, in_channels: int = 3,
                 hidden_size: int = 384, num_heads: int = 6, depth=(4, 4, 4),
                 bottleneck_dim: int = 64, in_context_len: int = 32,
                 skip: bool = True, skip_all_blocks: bool = True):
        super().__init__()
        self.input_size, self.patch_size = int(input_size), int(patch_size)
        self.in_channels, self.out_channels = int(in_channels), int(in_channels)
        self.hidden_size, self.num_heads = int(hidden_size), int(num_heads)
        self.depth = tuple(int(v) for v in depth)
        self.bottleneck_dim, self.in_context_len = int(bottleneck_dim), int(in_context_len)
        self.skip, self.skip_all_blocks = bool(skip), bool(skip_all_blocks)
        if self.depth != (4, 4, 4) or self.patch_size != 16:
            raise ValueError("PixelU-S-BEV preserves PixelU-B-16 depth=[4,4,4] and patch_size=16")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size)
        grid = input_size // patch_size
        pos = get_2d_sincos_pos_embed(hidden_size, grid).unsqueeze(0)
        self.register_buffer("pos_embed", pos, persistent=True)
        half_head = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(half_head, grid, 0)
        self.feat_rope_mid = VisionRotaryEmbeddingFast(half_head, grid // 2, in_context_len)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(half_head, grid, in_context_len)
        self.blocks0 = nn.ModuleList(JiTBlock(hidden_size, num_heads) for _ in range(self.depth[0]))
        self.blocks1 = nn.ModuleList(JiTBlock(hidden_size, num_heads) for _ in range(self.depth[1]))
        self.blocks2 = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, skip=skip),
            *[JiTBlock(hidden_size, num_heads, skip=skip_all_blocks if skip else False)
              for _ in range(self.depth[2] - 1)],
        ])
        self.downsampler = Downsample(hidden_size, hidden_size)
        self.upsampler = Upsample(hidden_size, hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(basic_init)
        for proj in (self.x_embedder.proj1, self.x_embedder.proj2):
            nn.init.xavier_uniform_(proj.weight.reshape(proj.weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj2.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=.02)
        for stage in (self.blocks0, self.blocks1, self.blocks2):
            for block in stage:
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        grid = int(tokens ** .5)
        if grid * grid != tokens:
            raise ValueError("Number of output image tokens must be square")
        x = x.reshape(batch, grid, grid, self.patch_size, self.patch_size, self.out_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self.out_channels, grid * self.patch_size, grid * self.patch_size
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                patch_condition: torch.Tensor | None = None,
                context_condition: torch.Tensor | None = None,
                patch_gate: torch.Tensor | float = 0.,
                context_gate: torch.Tensor | float = 0.) -> torch.Tensor:
        c = self.t_embedder(t)
        x = self.x_embedder(x)
        if patch_condition is not None:
            if patch_condition.shape != x.shape:
                raise RuntimeError(f"PixelU patch condition shape mismatch: {patch_condition.shape} vs {x.shape}")
            x = x + torch.as_tensor(patch_gate, dtype=x.dtype, device=x.device) * patch_condition
        x = x + self.pos_embed.to(x.dtype)
        skips = []
        for block in self.blocks0:
            x = block(x, c, self.feat_rope)
            skips.append(x)
        grid = self.input_size // self.patch_size
        x = self.downsampler(x, grid, grid)
        if self.in_context_len:
            if context_condition is None:
                context = x.new_zeros((x.shape[0], self.in_context_len, self.hidden_size))
            else:
                expected = (x.shape[0], self.in_context_len, self.hidden_size)
                if tuple(context_condition.shape) != expected:
                    raise RuntimeError(f"PixelU context condition shape mismatch: {context_condition.shape} vs {expected}")
                context = torch.as_tensor(context_gate, dtype=x.dtype, device=x.device) * context_condition
            x = torch.cat((context, x), dim=1)
        for block in self.blocks1:
            x = block(x, c, self.feat_rope_mid)
        context, image = (x[:, :self.in_context_len], x[:, self.in_context_len:]) if self.in_context_len else (None, x)
        image = self.upsampler(image, grid // 2, grid // 2)
        x = torch.cat((context, image), dim=1) if context is not None else image
        for block in self.blocks2:
            skip = skips.pop()
            skip = torch.cat((x[:, :self.in_context_len], skip), dim=1) if self.in_context_len else skip
            x = block(x, c, self.feat_rope_incontext if self.in_context_len else self.feat_rope, skip=skip)
        x = x[:, self.in_context_len:] if self.in_context_len else x
        return self.unpatchify(self.final_layer(x, c))
