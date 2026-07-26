# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared, torch-free tokenizer loader used by every backend."""

import json

import pytest

# transformers is a base dependency, but guard so the suite degrades
# gracefully if it is ever made optional.
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from transformers import (  # noqa: E402
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from marinfold.inference._tokenizer import (  # noqa: E402
    load_tokenizer,
    tokenizer_source_path,
)


def _write_checkpoint_tokenizer(directory):
    """Save a minimal WordLevel HF tokenizer the way a checkpoint ships it."""
    from tokenizers import Tokenizer, models

    inner = Tokenizer(
        models.WordLevel(
            vocab={"<pad>": 0, "<eos>": 1, "<UNK>": 2, "a": 3},
            unk_token="<UNK>",
        )
    )
    PreTrainedTokenizerFast(
        tokenizer_object=inner,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<UNK>",
    ).save_pretrained(str(directory))
    return directory


def test_load_tokenizer_uses_autotokenizer_for_standard_config(tmp_path) -> None:
    tok = load_tokenizer(_write_checkpoint_tokenizer(tmp_path))
    assert isinstance(tok, PreTrainedTokenizerFast)
    assert (tok.pad_token, tok.eos_token, tok.unk_token) == (
        "<pad>",
        "<eos>",
        "<UNK>",
    )


def test_load_tokenizer_falls_back_on_marinfold_custom_class(tmp_path) -> None:
    """Training-export checkpoints label the tokenizer ``TokenizersBackend``,
    which ``AutoTokenizer`` can't resolve; the loader must fall back to the
    shipped ``tokenizer.json``."""
    directory = _write_checkpoint_tokenizer(tmp_path)
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config))

    # Precondition: vanilla AutoTokenizer really does choke on this checkpoint.
    with pytest.raises(ValueError):
        AutoTokenizer.from_pretrained(str(directory))

    tok = load_tokenizer(directory)
    assert isinstance(tok, PreTrainedTokenizerFast)
    assert (tok.pad_token, tok.eos_token, tok.unk_token) == (
        "<pad>",
        "<eos>",
        "<UNK>",
    )
    ids = tok.encode("<eos>")
    assert tok.decode(ids) == "<eos>"


def test_load_tokenizer_reraises_without_tokenizer_json(tmp_path) -> None:
    """With no ``tokenizer.json`` to fall back to, the original load error
    must surface rather than a confusing secondary failure."""
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"})
    )
    with pytest.raises(Exception):
        load_tokenizer(tmp_path)


def test_tokenizer_source_path_passthrough_for_standard_config(tmp_path) -> None:
    """A checkpoint AutoTokenizer can load is handed to path-based loaders
    (e.g. vLLM) unchanged — no temp-dir repair."""
    directory = _write_checkpoint_tokenizer(tmp_path)
    assert tokenizer_source_path(directory) == str(directory)


def test_tokenizer_source_path_repairs_marinfold_custom_class(tmp_path) -> None:
    """A checkpoint AutoTokenizer *can't* load is repaired into a fresh dir
    that AutoTokenizer then loads cleanly (this is what vLLM consumes)."""
    directory = _write_checkpoint_tokenizer(tmp_path)
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config))

    repaired = tokenizer_source_path(directory)
    assert repaired != str(directory)
    # The repaired dir is loadable by the very API vLLM uses internally.
    tok = AutoTokenizer.from_pretrained(repaired)
    assert (tok.pad_token, tok.eos_token, tok.unk_token) == (
        "<pad>",
        "<eos>",
        "<UNK>",
    )
