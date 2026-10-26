"""Final model validation rejects corruption that a successful job can hide."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from experiments.exp364_models_mpnn_only_lr_sweep.verify_final import verify_export


def make_export(root: Path, value: float) -> None:
    save_file(
        {"weight": np.array([1, value], dtype=np.float32)}, root / "model.safetensors"
    )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {"weight": "model.safetensors"},
            }
        )
    )


def test_finite_export(tmp_path: Path) -> None:
    make_export(tmp_path, 2)
    result = verify_export(str(tmp_path))
    assert result["all_parameters_finite"]
    assert result["parameter_bytes"] == 8


def test_nonfinite_export(tmp_path: Path) -> None:
    make_export(tmp_path, float("nan"))
    with pytest.raises(ValueError, match="Nonfinite exported tensor"):
        verify_export(str(tmp_path))


def test_truncated_export(tmp_path: Path) -> None:
    make_export(tmp_path, 2)
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-4])
    with pytest.raises(ValueError, match="Truncated exported tensor"):
        verify_export(str(tmp_path))
