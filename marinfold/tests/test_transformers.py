# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from marinfold.inference._transformers import _resolve_dtype  # noqa: E402
from marinfold.inference._transformers import TransformersBackend  # noqa: E402
from marinfold import build_tokenizer  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402


def test_bfloat16_resolves_to_torch_bfloat16() -> None:
    assert _resolve_dtype("bfloat16") is torch.bfloat16


def test_shared_backend_default_dtype_stays_safe_for_mps_models() -> None:
    from marinfold.inference._transformers import TransformersBackend

    assert TransformersBackend.__init__.__kwdefaults__ == {
        "dtype": "bfloat16",
        "device": None,
        "tail_batch_size": 64,
    }


def test_transformers_reexports_shared_tokenizer_loader() -> None:
    """The backend still exposes ``_load_tokenizer`` for existing callers,
    now aliased to the shared implementation."""
    from marinfold.inference._tokenizer import load_tokenizer
    from marinfold.inference._transformers import _load_tokenizer

    assert _load_tokenizer is load_tokenizer


def test_sample_minimum_suppresses_preferred_eos_only_for_new_tokens(tmp_path) -> None:
    tokenizer = build_tokenizer(["<start>", "<end>", "<other>"])
    stop_id = tokenizer.convert_tokens_to_ids("<end>")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=32, n_embd=32, n_layer=1, n_head=1,
        eos_token_id=stop_id, bos_token_id=None,
    ))
    # A real CPU model with constant logits: EOS overwhelmingly dominates,
    # so the constrained path must mask it before sampling.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.ln_f.bias.fill_(1.0)
        model.lm_head.weight[stop_id].fill_(1.0)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    backend = TransformersBackend(tmp_path, device="cpu", dtype="float32")
    prompts = [[tokenizer.convert_tokens_to_ids("<start>")] * 10] * 2
    common = dict(max_new_tokens=8, stop_token_id=stop_id, seed=42, batch_size=1)
    assert backend.sample_completions(prompts, **common) == [[], []]
    constrained = backend.sample_completions(prompts, min_new_tokens=3, **common)
    assert [len(row) for row in constrained] == [3, 3]
    assert all(stop_id not in row for row in constrained)
    with pytest.raises(ValueError, match="min_new_tokens"):
        backend.sample_completions(prompts, min_new_tokens=9, **common)
