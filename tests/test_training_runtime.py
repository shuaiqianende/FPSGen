import random

import numpy as np
import pytest
import torch

from fpsgen.utils.training_runtime import (
    apply_training_environment,
    load_training_config,
    seed_training,
)


def test_seed_training_repeats_python_numpy_and_torch_streams():
    seed_training(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    seed_training(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert first == second


def test_load_training_config_requires_mapping(tmp_path):
    valid = tmp_path / "valid.yaml"
    valid.write_text("data:\n  data_dir: ''\n", encoding="utf-8")
    assert load_training_config(valid) == {"data": {"data_dir": ""}}

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_training_config(invalid)


def test_apply_training_environment_overrides_only_data_root():
    config = {"data": {"data_dir": "old"}, "train": {"batch_size": 8}}
    result = apply_training_environment(config, {"TRAIN_DATABASE": "/dataset"})
    assert result["data"]["data_dir"] == "/dataset"
    assert result["train"] == {"batch_size": 8}
