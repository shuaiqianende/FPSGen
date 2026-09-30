"""Shared, dependency-light helpers for FPSGen training entry points."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch
import yaml


def seed_training(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch consistently for one training process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_training_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping and fail early for empty or malformed configs."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(
            f"Training config must be a YAML mapping, got {type(config).__name__}: "
            f"{config_path}"
        )
    return config


def apply_training_environment(
    config: MutableMapping[str, Any], environment: Mapping[str, str] | None = None
) -> MutableMapping[str, Any]:
    """Apply portable runtime overrides without storing local paths in YAML."""
    environment = os.environ if environment is None else environment
    data_root = environment.get("TRAIN_DATABASE")
    if data_root:
        data = config.get("data")
        if not isinstance(data, MutableMapping):
            raise ValueError("TRAIN_DATABASE was set but config.data is not a mapping")
        data["data_dir"] = data_root
    return config
