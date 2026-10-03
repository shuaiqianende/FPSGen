"""Safe configuration parsing shared by training and repository tooling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate explicit keys while allowing standard YAML merge overrides."""

    def construct_mapping(self, node, deep=False):
        explicit_keys = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in explicit_keys
                explicit_keys.add(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while reading configuration", node.start_mark,
                    "mapping keys must be hashable", key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while reading configuration", node.start_mark,
                    f"duplicate configuration key: {key!r}", key_node.start_mark,
                )
        return super().construct_mapping(node, deep=deep)


def load_yaml_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a non-empty YAML mapping with source locations on parse errors."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=UniqueKeySafeLoader)
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Configuration must be a non-empty YAML mapping: {config_path}")
    return config
