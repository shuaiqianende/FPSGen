"""PixelDiT-S-BEV raw-pixel Patch-DiT plus Pixel-level Transformer core.

The topology follows NVlabs/PixelDiT (commit 41f73006): patch semantic
tokens are processed by DiT blocks and each 16x16 patch is reconstructed by
PiT blocks.  This is a width-scaled FPSGen variant, not an NVIDIA release.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pixelu_core import Attention, RMSNorm, SwiGLUFFN, TimestepEmbedder, VisionRotaryEmbeddingFast, modulate


class AugmentedDiTBlock(nn.Module):
    """AdaLN DiT block for the patch-semantic level."""

    def __init__(self, hidden_size: int, groups: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, groups, qkv_bias=False, qk_norm=True)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, hidden_size * 4, bias=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(context).chunk(6, dim=-1)
        tokens = tokens + gate_a[:, None] * self.attn(modulate(self.norm1(tokens), shift_a, scale_a), rope)
        return tokens + gate_m[:, None] * self.mlp(modulate(self.norm2(tokens), shift_m, scale_m))


class PiTBlock(nn.Module):
    """Pixel-level transformer that preserves the official post-modulation path."""

    def __init__(self, pixel_hidden_size: int, semantic_size: int, patch_pixels: int,
                 post_modulation: bool):
        super().__init__()
        self.pixel_hidden_size, self.semantic_size = int(pixel_hidden_size), int(semantic_size)
        self.patch_pixels, self.post_modulation = int(patch_pixels), bool(post_modulation)
        self.norm1 = RMSNorm(pixel_hidden_size)
        self.norm2 = RMSNorm(pixel_hidden_size)
        # PiT must contain an actual pixel-level Transformer. Attention is
        # local to each 16x16 patch; Patch-DiT carries global semantics.
        self.attn = Attention(pixel_hidden_size, 1, qkv_bias=False, qk_norm=True)
        self.pixel_mlp = SwiGLUFFN(pixel_hidden_size, pixel_hidden_size * 4, bias=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(semantic_size, pixel_hidden_size * 4))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, pixels: torch.Tensor, semantic: torch.Tensor,
                rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        # pixels [B, L, P, C], semantic [B, L, D]. Global mixing happens in
        # Patch-DiT; PiT restores spatial detail within each patch.
        b, l, p, c = pixels.shape
        if (p, c) != (self.patch_pixels, self.pixel_hidden_size):
            raise ValueError("PixelDiT PiT token shape mismatch")
        shift1, scale1, shift2, scale2 = self.ada(semantic).view(b, l, 4, 1, c).unbind(2)
        first = self.norm1(pixels)
        if not self.post_modulation:
            first = first * (1 + scale1) + shift1
        mixed = self.attn(first.reshape(b * l, p, c), rope).view(b, l, p, c)
        pixels = pixels + (mixed * (1 + scale1) + shift1 if self.post_modulation else mixed)
        second = self.norm2(pixels)
        if not self.post_modulation:
            second = second * (1 + scale2) + shift2
        update = self.pixel_mlp(second)
        return pixels + (update * (1 + scale2) + shift2 if self.post_modulation else update)


class PixelDiTCore(nn.Module):
    """Raw-pixel dual-level PixelDiT with direct velocity output."""

    def __init__(self, input_size=256, patch_size=16, in_channels=3, hidden_size=384,
                 num_groups=6, patch_depth=8, pixel_hidden_size=8, pixel_depth=4,
                 pit_adaln_post_modulation=True, time_scale=1.0, repa=False):
        super().__init__()
        self.input_size, self.patch_size, self.in_channels = int(input_size), int(patch_size), int(in_channels)
        self.hidden_size, self.num_groups, self.patch_depth = int(hidden_size), int(num_groups), int(patch_depth)
        self.pixel_hidden_size, self.pixel_depth = int(pixel_hidden_size), int(pixel_depth)
        self.pit_adaln_post_modulation, self.time_scale, self.repa = bool(pit_adaln_post_modulation), float(time_scale), bool(repa)
        if self.input_size % self.patch_size or self.patch_size != 16 or self.hidden_size // self.num_groups != 64:
            raise ValueError("PixelDiT-S-BEV requires patch16 and head dimension 64")
        if self.pixel_depth <= 0 or self.repa:
            raise ValueError("PixelDiT-S-BEV requires PiT depth > 0 and REPA disabled")
        self.grid, self.num_patches, self.patch_pixels = self.input_size // self.patch_size, (self.input_size // self.patch_size) ** 2, self.patch_size ** 2
        self.patch_embed = nn.Linear(self.in_channels * self.patch_pixels, self.hidden_size)
        self.pixel_embed = nn.Linear(self.in_channels, self.pixel_hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.patch_rope = VisionRotaryEmbeddingFast(self.hidden_size // self.num_groups // 2, self.grid, 0)
        self.pixel_rope = VisionRotaryEmbeddingFast(self.pixel_hidden_size // 2, self.patch_size, 0)
        self.patch_blocks = nn.ModuleList(AugmentedDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.patch_depth))
        self.pixel_blocks = nn.ModuleList(PiTBlock(self.pixel_hidden_size, self.hidden_size, self.patch_pixels, self.pit_adaln_post_modulation) for _ in range(self.pixel_depth))
        self.pixel_out = nn.Linear(self.pixel_hidden_size, self.in_channels)
        nn.init.zeros_(self.pixel_out.weight)
        nn.init.zeros_(self.pixel_out.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size} input")
        b, c, _, _ = x.shape
        return x.reshape(b, c, self.grid, self.patch_size, self.grid, self.patch_size).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, self.patch_pixels, c)

    def unpatchify(self, pixels: torch.Tensor) -> torch.Tensor:
        b, l, p, c = pixels.shape
        if (l, p, c) != (self.num_patches, self.patch_pixels, self.in_channels):
            raise ValueError("PixelDiT output patch shape mismatch")
        return pixels.reshape(b, self.grid, self.grid, self.patch_size, self.patch_size, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, self.input_size, self.input_size)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, pixel_condition: torch.Tensor | None = None,
                patch_condition: torch.Tensor | None = None, global_condition: torch.Tensor | None = None,
                pixel_gate=0., patch_gate=0., global_gate=0., patch_adapter=None,
                control_tokens: torch.Tensor | None = None) -> torch.Tensor:
        pixels = self.patchify(xt)
        tokens = self.patch_embed(pixels.flatten(2))
        if patch_condition is not None:
            if patch_condition.shape != tokens.shape: raise ValueError("PixelDiT patch condition shape mismatch")
            tokens = tokens + torch.as_tensor(patch_gate, dtype=tokens.dtype, device=tokens.device) * patch_condition
        context = self.t_embedder(t * self.time_scale)
        if global_condition is not None:
            if global_condition.shape != context.shape: raise ValueError("PixelDiT global condition shape mismatch")
            context = F.silu(context + torch.as_tensor(global_gate, dtype=context.dtype, device=context.device) * global_condition)
        if patch_adapter is not None and control_tokens is None:
            raise ValueError("PixelControl adapter requires patch-aligned control tokens")
        for index, block in enumerate(self.patch_blocks):
            tokens = block(tokens, context, self.patch_rope)
            if patch_adapter is not None:
                tokens = tokens + patch_adapter.residual(index, control_tokens, tokens)
        # A semantic residual keeps patch-level condition available to every PiT block.
        tokens = F.silu(tokens + context[:, None])
        pixel_tokens = self.pixel_embed(pixels)
        if pixel_condition is not None:
            if pixel_condition.shape != pixel_tokens.shape: raise ValueError("PixelDiT pixel condition shape mismatch")
            pixel_tokens = pixel_tokens + torch.as_tensor(pixel_gate, dtype=pixel_tokens.dtype, device=pixel_tokens.device) * pixel_condition
        for block in self.pixel_blocks:
            pixel_tokens = block(pixel_tokens, tokens, self.pixel_rope)
        return self.unpatchify(self.pixel_out(pixel_tokens))
