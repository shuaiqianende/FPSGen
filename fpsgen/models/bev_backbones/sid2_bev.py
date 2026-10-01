"""SiD2-S-BEV wrapper with separate-first spatial/global conditions."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .condition import _require_bchw, modality_norms_and_ratios
from .sid2_core import SiD2Core


class SiD2ConditionEncoder(nn.Module):
    def __init__(self, channels=(64,128,256,384), time_dim=256, lidar_channels=32):
        super().__init__(); self.channels=tuple(channels); self.lidar_channels=int(lidar_channels)
        self.lidar = nn.Conv2d(lidar_channels, channels[0], 2, stride=2, bias=False)
        self.vehicle = nn.Conv2d(1, channels[0], 2, stride=2, bias=False); self.road = nn.Conv2d(1, channels[0], 2, stride=2, bias=False)
        self.down = nn.ModuleList(nn.Conv2d(channels[i], channels[i+1], 1, bias=False) for i in range(3))
        self.global_proj = nn.ModuleList(nn.Linear(width, time_dim, bias=False) for width in channels); self._norms={}
    def forward(self, raw, layout):
        _require_bchw(raw, layout, self.lidar_channels+2)
        parts={"lidar":self.lidar(raw), "vehicle":self.vehicle(layout[:,:1]), "road":self.road(layout[:,1:2])}; maps=[sum(parts.values())]
        self._norms=modality_norms_and_ratios(parts, channel_dim=1)
        for layer in self.down: maps.append(layer(F.avg_pool2d(maps[-1],2)))
        return tuple(maps), tuple(p(m.mean((2,3))) for p,m in zip(self.global_proj,maps))
    def modality_norms(self): return self._norms


class BEVSiD2S(nn.Module):
    def __init__(self, model_cfg):
        super().__init__(); from fpsgen.models.image_flow_net import DynamicKNNPillarEncoder
        cfg, cond = model_cfg.get("sid2", {}), model_cfg.get("condition", {}); self.input_size=int(cfg.get("input_size",256)); channels=tuple(cfg.get("channels",[64,128,256,384])); time_dim=int(cfg.get("time_dim",256))
        self.pc_encoder=DynamicKNNPillarEncoder(3,int(cond.get("lidar_channels",32)),self.input_size)
        self.condition_encoder=SiD2ConditionEncoder(channels,time_dim,int(cond.get("lidar_channels",32)))
        self.core=SiD2Core(self.input_size,int(cfg.get("patch_size",2)),int(cfg.get("in_channels",3)),channels,tuple(cfg.get("num_updown_blocks",[3,3,3])),int(cfg.get("num_mid_blocks",16)),int(cfg.get("head_dim",64)),time_dim)
        self.spatial_gates=nn.Parameter(torch.full((4,),float(cond.get("gate_init",.1)))); self.global_gates=nn.Parameter(torch.full((4,),float(cond.get("gate_init",.1)))); self._norms={}
    def get_raw_pc_bev(self, points): return self.pc_encoder(points)
    def forward(self, xt,t,raw_pc,layout):
        maps, glob=self.condition_encoder(raw_pc,layout); spatial=tuple(g.to(xt.dtype)*m for g,m in zip(self.spatial_gates,maps)); global_values=tuple(g.to(xt.dtype)*v for g,v in zip(self.global_gates,glob)); self._norms={"spatial_norm":maps[0].detach().float().norm(dim=1).mean(),"global_norm":glob[0].detach().float().norm(dim=-1).mean()}; return self.core(xt,t,spatial,global_values)
    def condition_diagnostics(self): return {**{f"gate_spatial_{i}":g.detach() for i,g in enumerate(self.spatial_gates)},**{f"gate_global_{i}":g.detach() for i,g in enumerate(self.global_gates)},**self._norms,**self.condition_encoder.modality_norms()}
