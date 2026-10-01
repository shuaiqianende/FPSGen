"""Shared, explicit spatial-condition primitives for Stage-1 BEV studies."""

from .condition_packet import ConditionPacket, make_condition_packet
from .lite_spade import LiteSPADE
from .pixelcontrol_adapter import PixelControlAdapter, PixelControlEncoder

__all__ = [
    "ConditionPacket", "make_condition_packet", "LiteSPADE",
    "PixelControlAdapter", "PixelControlEncoder",
]
