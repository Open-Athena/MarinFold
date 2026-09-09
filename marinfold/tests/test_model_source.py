# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for checkpoint overlays consumed by path-based model loaders."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from transformers import PreTrainedTokenizerFast  # noqa: E402

from marinfold.inference._model_source import model_source_path  # noqa: E402


def _write_fixed_position_checkpoint(
    directory: Path,
    *,
    vocab_size: int,
    start_token_id: int,
    num_tokens: int,
    token_embeddings: np.ndarray,
    include_position_config: bool = True,
) -> None:
    """Write a tiny fixed-residue-position checkpoint export."""

    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models

    vocab = {"<pad>": 0, "<eos>": 1, "<UNK>": 2}
    position_ids = set(range(start_token_id, start_token_id + num_tokens))
    for token_id in range(3, vocab_size):
        if token_id not in position_ids:
            vocab[f"x{token_id}"] = token_id
    for offset in range(num_tokens):
        vocab[f"<p{offset}>"] = start_token_id + offset
    inner = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    PreTrainedTokenizerFast(
        tokenizer_object=inner,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<UNK>",
    ).save_pretrained(str(directory))

    config = {
        "model_type": "qwen3",
        "vocab_size": vocab_size,
    }
    if include_position_config:
        config["position_embedding"] = {
            "start_token_id": start_token_id,
            "num_tokens": num_tokens,
            "base": 10_000.0,
        }
    (directory / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "token_embeddings.weight": token_embeddings,
            "model.norm.weight": np.ones((token_embeddings.shape[1],), dtype=np.float32),
        },
        directory / "model-00001-of-00001.safetensors",
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int(token_embeddings.nbytes)},
                "weight_map": {
                    "token_embeddings.weight": "model-00001-of-00001.safetensors",
                    "model.norm.weight": "model-00001-of-00001.safetensors",
                },
            }
        )
    )


def _fixed_rows(num_tokens: int, embed_dim: int) -> np.ndarray:
    positions = np.arange(num_tokens, dtype=np.float32)[:, None]
    channels = np.arange(embed_dim // 2, dtype=np.float32)
    inv_freq = 10_000.0 ** (-2.0 * channels / embed_dim)
    angles = positions * inv_freq[None, :]
    return np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(
        num_tokens,
        embed_dim,
    )


def test_model_source_path_materializes_compact_fixed_position_embeddings(tmp_path) -> None:
    """Compact exports get fixed rows inserted into a standard HF table."""

    from safetensors.numpy import load_file

    learned = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    _write_fixed_position_checkpoint(
        tmp_path,
        vocab_size=8,
        start_token_id=3,
        num_tokens=2,
        token_embeddings=learned,
    )

    repaired = Path(model_source_path(tmp_path))
    tensors = load_file(repaired / "model-00001-of-00001.safetensors")
    index = json.loads((repaired / "model.safetensors.index.json").read_text())

    assert repaired != tmp_path
    assert "token_embeddings.weight" not in tensors
    assert index["weight_map"]["model.embed_tokens.weight"] == "model-00001-of-00001.safetensors"
    full = tensors["model.embed_tokens.weight"]
    np.testing.assert_array_equal(full[:3], learned[:3])
    np.testing.assert_allclose(full[3:5], _fixed_rows(2, 4), rtol=1e-6)
    np.testing.assert_array_equal(full[5:], learned[3:])
    original = load_file(tmp_path / "model-00001-of-00001.safetensors")
    assert "token_embeddings.weight" in original


def test_model_source_path_materializes_full_delta_position_embeddings(tmp_path) -> None:
    """Full-vocab delta exports add fixed rows on the configured span."""

    from safetensors.numpy import load_file

    learned = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    _write_fixed_position_checkpoint(
        tmp_path,
        vocab_size=8,
        start_token_id=3,
        num_tokens=2,
        token_embeddings=learned,
    )

    repaired = Path(model_source_path(tmp_path))
    full = load_file(repaired / "model-00001-of-00001.safetensors")[
        "model.embed_tokens.weight"
    ]

    np.testing.assert_array_equal(full[:3], learned[:3])
    np.testing.assert_allclose(full[3:5], learned[3:5] + _fixed_rows(2, 4), rtol=1e-6)
    np.testing.assert_array_equal(full[5:], learned[5:])


def test_model_source_path_compact_shape_bounds_extra_tokenizer_positions(tmp_path) -> None:
    """Compact shape determines how many inferred position tokens were removed."""

    from safetensors.numpy import load_file

    learned = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    _write_fixed_position_checkpoint(
        tmp_path,
        vocab_size=8,
        start_token_id=3,
        num_tokens=3,
        token_embeddings=learned,
        include_position_config=False,
    )

    repaired = Path(model_source_path(tmp_path))
    full = load_file(repaired / "model-00001-of-00001.safetensors")[
        "model.embed_tokens.weight"
    ]

    np.testing.assert_array_equal(full[:3], learned[:3])
    np.testing.assert_allclose(full[3:5], _fixed_rows(2, 4), rtol=1e-6)
    np.testing.assert_array_equal(full[5:], learned[3:])


def test_model_source_path_auto_renames_unmarked_full_embedding_table(tmp_path) -> None:
    """Unmarked full-vocab Levanter tables are not assumed to be deltas."""

    from safetensors.numpy import load_file

    learned = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    _write_fixed_position_checkpoint(
        tmp_path,
        vocab_size=8,
        start_token_id=3,
        num_tokens=2,
        token_embeddings=learned,
        include_position_config=False,
    )

    repaired = Path(model_source_path(tmp_path))
    full = load_file(repaired / "model-00001-of-00001.safetensors")[
        "model.embed_tokens.weight"
    ]

    np.testing.assert_array_equal(full, learned)


def test_model_source_path_force_adds_fixed_rows_to_unmarked_full_delta_table(tmp_path) -> None:
    """Force mode supports older full-delta exports that omitted metadata."""

    from safetensors.numpy import load_file

    learned = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    _write_fixed_position_checkpoint(
        tmp_path,
        vocab_size=8,
        start_token_id=3,
        num_tokens=2,
        token_embeddings=learned,
        include_position_config=False,
    )

    repaired = Path(model_source_path(tmp_path, fixed_residue_position_embeddings="force"))
    full = load_file(repaired / "model-00001-of-00001.safetensors")[
        "model.embed_tokens.weight"
    ]

    np.testing.assert_array_equal(full[:3], learned[:3])
    np.testing.assert_allclose(full[3:5], learned[3:5] + _fixed_rows(2, 4), rtol=1e-6)
    np.testing.assert_array_equal(full[5:], learned[5:])
