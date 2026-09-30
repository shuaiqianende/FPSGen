"""DiP-S/16 pixel-space Flow-Matching core for FPSGen.

Architectural source: ``NJU-PCALab/DiP`` commit
``949294f290f2380a7767649f343679e24dab4410``, especially its flattened
global transformer and patch-local U-Net Detailer.  This is an independent
architectural implementation; DiP-S-BEV is an FPSGen width-scaled variant,
not an upstream model name.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pixelu_core import RMSNorm, Attention, SwiGLUFFN, TimestepEmbedder, VisionRotaryEmbeddingFast, modulate


class FlattenDiTBlock(nn.Module):
    """DiP-style global flattened-patch block with AdaLN, QK norm and RoPE."""

    def __init__(self, hidden_size: int, num_groups: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_groups, qkv_bias=True, qk_norm=True)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, hidden_size * 4)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_a[:, None] * self.attn(modulate(self.norm1(x), shift_a, scale_a), rope)
        return x + gate_m[:, None] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class LocalDetailer(nn.Module):
    """Patch-local 16→8→4→2→1→2→4→8→16 U-Net Detailer."""

    def __init__(self, patch_size: int, in_channels: int, hidden_size: int, channels=(64, 128, 256, 512)):
        super().__init__()
        if patch_size != 16:
            raise ValueError("DiP-S-BEV preserves the DiP-S/16 local-detail patch size")
        c0, c1, c2, c3 = (int(v) for v in channels)
        self.patch_size, self.channels = int(patch_size), (c0, c1, c2, c3)
        self.down0 = nn.Sequential(nn.Conv2d(in_channels, c0, 3, padding=1), nn.SiLU(), nn.Conv2d(c0, c0, 3, padding=1), nn.SiLU())
        self.down1 = nn.Sequential(nn.Conv2d(c0, c1, 4, stride=2, padding=1), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.SiLU())
        self.down3 = nn.Sequential(nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.SiLU())
        self.down4 = nn.Sequential(nn.Conv2d(c3, c3, 4, stride=2, padding=1), nn.SiLU())
        self.global_inject = nn.Linear(hidden_size, c3)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(c3, c3, 4, stride=2, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(c3 + c3, c2, 4, stride=2, padding=1), nn.SiLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(c2 + c2, c1, 4, stride=2, padding=1), nn.SiLU())
        self.up0 = nn.Sequential(nn.ConvTranspose2d(c1 + c1, c0, 4, stride=2, padding=1), nn.SiLU())
        self.out = nn.Sequential(nn.Conv2d(c0 + c0, c0, 3, padding=1), nn.SiLU(), nn.Conv2d(c0, in_channels, 3, padding=1))

    def forward(self, patches: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = patches.shape
        if (h, w) != (self.patch_size, self.patch_size):
            raise ValueError(f"LocalDetailer expects {self.patch_size}x{self.patch_size} patches")
        x = patches.reshape(b * n, c, h, w)
        token = tokens.reshape(b * n, -1)
        e0 = self.down0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        mid = self.down4(e3) + self.global_inject(token)[:, :, None, None]
        d3 = self.up3(mid)
        d2 = self.up2(torch.cat((d3, e3), dim=1))
        d1 = self.up1(torch.cat((d2, e2), dim=1))
        d0 = self.up0(torch.cat((d1, e1), dim=1))
        out = self.out(torch.cat((d0, e0), dim=1))
        return out.reshape(b, n, c, h, w)


class DiPCore(nn.Module):
    """Global flattened transformer plus the mandatory local patch Detailer."""

    def __init__(self, input_size=256, patch_size=16, in_channels=3, hidden_size=384,
                 num_groups=6, num_cond_blocks=8, local_channels=(64, 128, 256, 512)):
        super().__init__()
        self.input_size, self.patch_size, self.in_channels = int(input_size), int(patch_size), int(in_channels)
        self.hidden_size, self.num_groups, self.num_cond_blocks = int(hidden_size), int(num_groups), int(num_cond_blocks)
        self.local_channels = tuple(map(int, local_channels))
        if self.input_size % self.patch_size:
            raise ValueError("input_size must divide patch_size")
        if self.hidden_size % self.num_groups or self.hidden_size // self.num_groups != 64:
            raise ValueError("DiP-S-BEV keeps head dimension 64")
        self.grid = self.input_size // self.patch_size
        self.num_patches = self.grid * self.grid
        self.patch_embed = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.global_condition_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=False)
        self.rope = VisionRotaryEmbeddingFast(self.hidden_size // self.num_groups // 2, self.grid, 0)
        self.blocks = nn.ModuleList(FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_cond_blocks))
        self.detailer = LocalDetailer(self.patch_size, self.in_channels, self.hidden_size, self.local_channels)
        self._init_weights()

    def _init_weights(self):
        grid = torch.arange(self.grid, dtype=torch.float32)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        freq = torch.arange(self.hidden_size // 4, dtype=torch.float32) / (self.hidden_size // 4)
        omega = 1.0 / (10000 ** freq)
        pos = torch.cat((torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega),
                         torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega)), dim=1)
        self.pos_embed.data.copy_(pos.unsqueeze(0))
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != (self.input_size, self.input_size):
            raise ValueError(f"Expected {self.input_size}x{self.input_size}, got {(h, w)}")
        return x.reshape(b, c, self.grid, self.patch_size, self.grid, self.patch_size).permute(0, 2, 4, 1, 3, 5).reshape(b, self.num_patches, c, self.patch_size, self.patch_size)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, c, ph, pw = patches.shape
        if n != self.num_patches:
            raise ValueError("Wrong number of DiP patches")
        return patches.reshape(b, self.grid, self.grid, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, self.input_size, self.input_size)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, patch_condition: torch.Tensor | None = None,
                global_condition: torch.Tensor | None = None) -> torch.Tensor:
        patches = self.patchify(xt)
        tokens = self.patch_embed(patches.flatten(2)) + self.pos_embed.to(xt.dtype)
        if patch_condition is not None:
            if patch_condition.shape != tokens.shape:
                raise ValueError(f"DiP patch condition mismatch {tuple(patch_condition.shape)} vs {tuple(tokens.shape)}")
            tokens = tokens + patch_condition
        context = self.t_embedder(t)
        if global_condition is not None:
            context = F.silu(context + self.global_condition_proj(global_condition))
        for block in self.blocks:
            tokens = block(tokens, context, self.rope)
        return self.unpatchify(self.detailer(patches, tokens))
