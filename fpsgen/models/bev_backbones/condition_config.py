"""Configuration contract shared by all Stage-1 condition adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionOptions:
    spatial: bool = True
    global_modulation: bool = True
    fusion: str = "shared"
    native: bool = False
    gate_init: float = 0.1
    api_version: int = 1


def condition_options(model_cfg) -> ConditionOptions:
    """Parse the backward-compatible condition API and validate hard contracts."""
    raw = model_cfg.get("condition", {})
    fusion = str(raw.get("fusion", "shared")).lower()
    if fusion not in {"shared", "separate"}:
        raise ValueError("model.condition.fusion must be 'shared' or 'separate'")
    if bool(raw.get("bias", False)):
        raise ValueError("Condition projections must be bias-free to preserve f(0)=0")
    gate_init = float(raw.get("gate_init", 0.1))
    if gate_init <= 0:
        raise ValueError("model.condition.gate_init must be positive")
    return ConditionOptions(
        spatial=bool(raw.get("spatial", True)),
        global_modulation=bool(raw.get("global", True)),
        fusion=fusion,
        native=bool(raw.get("native", False)),
        gate_init=gate_init,
        api_version=int(raw.get("api_version", 1)),
    )
