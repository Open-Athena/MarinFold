# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared, torch-free model-config loader.

The failure these guard against is silent: transformers 4.x does not error on
a transformers-5 ``rope_parameters`` block, it ignores it and uses the
architecture's default rope. So every assertion here is about *values*, not
about whether loading raised.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from marinfold.inference._config import (  # noqa: E402
    load_config,
    needs_rope_repair,
    read_config,
    repair_rope,
)

# The rope block a levanter checkpoint exported by transformers 5.12.1 writes
# for `Llama3RotaryEmbeddingsConfig()` — verbatim from the #117 checkpoint.
TF5_ROPE = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
    "rope_theta": 500000,
}


def _write_config(directory: Path, **overrides) -> Path:
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 2845,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 8192,
    }
    config.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def test_transformers5_export_is_detected(tmp_path):
    assert needs_rope_repair(read_config(_write_config(tmp_path, rope_parameters=TF5_ROPE)))


def test_transformers4_export_is_left_alone(tmp_path):
    raw = read_config(
        _write_config(tmp_path, rope_theta=500000, rope_scaling={"rope_type": "llama3"})
    )
    assert not needs_rope_repair(raw)
    assert repair_rope(raw) is raw


def test_export_carrying_both_shapes_is_left_alone(tmp_path):
    """A previously repaired directory must not be repaired again."""
    raw = read_config(
        _write_config(tmp_path, rope_parameters=TF5_ROPE, rope_theta=500000,
                      rope_scaling={"rope_type": "llama3"})
    )
    assert not needs_rope_repair(raw)


def test_no_rope_block_at_all(tmp_path):
    assert not needs_rope_repair(read_config(_write_config(tmp_path)))


def test_missing_config_is_not_an_error(tmp_path):
    assert read_config(tmp_path) == {}
    assert not needs_rope_repair({})


def test_repair_moves_theta_out_and_scaling_across(tmp_path):
    repaired = repair_rope(read_config(_write_config(tmp_path, rope_parameters=TF5_ROPE)))
    assert repaired["rope_theta"] == 500000
    assert repaired["rope_scaling"]["rope_type"] == "llama3"
    assert repaired["rope_scaling"]["factor"] == 8.0
    # rope_theta belongs at the top level, not inside the scaling spec.
    assert "rope_theta" not in repaired["rope_scaling"]
    # The 5.x block is kept so a repaired directory still loads under 5.x.
    assert repaired["rope_parameters"] == TF5_ROPE


def test_repair_does_not_mutate_its_input(tmp_path):
    raw = read_config(_write_config(tmp_path, rope_parameters=TF5_ROPE))
    before = json.dumps(raw, sort_keys=True)
    repair_rope(raw)
    assert json.dumps(raw, sort_keys=True) == before


def test_unscaled_rope_type_yields_no_scaling(tmp_path):
    raw = read_config(
        _write_config(tmp_path, rope_parameters={"rope_type": "default",
                                                 "rope_theta": 10_000})
    )
    repaired = repair_rope(raw)
    assert repaired["rope_theta"] == 10_000
    assert repaired["rope_scaling"] is None


def test_loaded_config_carries_the_trained_rope(tmp_path):
    """The end the bug actually shows up at: the values transformers reads.

    Without the repair this loads theta 10000 and no scaling — a 50x error in
    the rope base, applied silently.
    """
    config = load_config(_write_config(tmp_path, rope_parameters=TF5_ROPE))
    assert config.rope_theta == 500000
    assert config.rope_scaling is not None
    assert config.rope_scaling["rope_type"] == "llama3"
    assert config.rope_scaling["factor"] == 8.0
    # Architecture fields must survive the round-trip through for_model().
    assert config.vocab_size == 2845
    assert config.num_hidden_layers == 2
    assert config.num_key_value_heads == 2


def test_loaded_config_matches_autoconfig_when_no_repair_needed(tmp_path):
    from transformers import AutoConfig

    directory = _write_config(tmp_path, rope_theta=500000)
    assert load_config(directory).to_dict() == AutoConfig.from_pretrained(
        str(directory)
    ).to_dict()
