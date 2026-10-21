"""Standalone exports preserve tensor identities and nonfloating buffers."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from export_eval_checkpoint import convert_shard


def test_shard_round_trip_preserves_names_shapes_and_integer_values(
    tmp_path: Path,
) -> None:
    weights = {
        "model.weight": torch.randn(5, 7),
        "model.counter": torch.tensor([123456789], dtype=torch.int64),
    }
    source, destination = tmp_path / "source", tmp_path / "destination"
    save_file(weights, source)
    size = convert_shard(source, destination)
    restored = load_file(destination)
    assert restored.keys() == weights.keys()
    torch.testing.assert_close(
        restored["model.weight"], weights["model.weight"].bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored["model.counter"], weights["model.counter"], rtol=0, atol=0
    )
    assert size == 5 * 7 * 2 + 8
