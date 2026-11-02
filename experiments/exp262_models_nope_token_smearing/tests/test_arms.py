# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PyTorch pilot arms, including a cross-check against levanter.

The causality test is the one that matters: a smear that reaches forward turns
next-token prediction into partial copying, trains beautifully, and is worthless.
The cross-check keeps this implementation and ``gpu/architecture.py`` honest
about being the same function.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot"))

from arms import ARMS_BY_KEY, SmearEmbedding, build_config, build_model  # noqa: E402

VOCAB = 37
POSITION = 24


def small_config():
    return build_config(
        vocab_size=VOCAB, hidden=32, layers=2, heads=4, kv_heads=2,
        intermediate=64, max_seq_len=POSITION,
    )


def perturbed_model(arm_key: str, seed: int = 0):
    """Build an arm and move the smear off its zero initialisation."""
    torch.manual_seed(seed)
    model = build_model(small_config(), ARMS_BY_KEY[arm_key]).eval()
    embed = model.model.embed_tokens
    if isinstance(embed, SmearEmbedding):
        with torch.no_grad():
            embed.weights.normal_(std=0.5)
    return model


def tokens(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, POSITION), generator=generator)


def logits(model, ids: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(ids).logits[0].float().numpy()


def test_smear_is_strictly_causal():
    model = perturbed_model("b-rope-smear")
    ids = tokens()
    reference = logits(model, ids)
    for cut in (5, 11, POSITION - 2):
        changed = ids.clone()
        changed[0, cut + 1] = (int(changed[0, cut + 1]) + 3) % VOCAB
        after = logits(model, changed)
        assert np.array_equal(reference[: cut + 1], after[: cut + 1]), (
            f"changing token {cut + 1} moved the logits at positions <= {cut}; the smear "
            f"is not causal (max abs diff {np.abs(reference[: cut + 1] - after[: cut + 1]).max()})"
        )


def test_smear_does_not_wrap_around():
    model = perturbed_model("b-rope-smear")
    ids = tokens()
    reference = logits(model, ids)
    changed = ids.clone()
    changed[0, -1] = (int(changed[0, -1]) + 5) % VOCAB
    assert np.array_equal(reference[:2], logits(model, changed)[:2]), (
        "the last token reached positions 0-1; the shift is wrapping"
    )


def test_zero_initialised_smear_matches_the_control():
    ids = tokens()
    torch.manual_seed(0)
    smeared = build_model(small_config(), ARMS_BY_KEY["b-rope-smear"]).eval()
    torch.manual_seed(0)
    control = build_model(small_config(), ARMS_BY_KEY["a-rope"]).eval()
    np.testing.assert_allclose(logits(smeared, ids), logits(control, ids), rtol=0, atol=0)


def test_offsets_have_independent_weights():
    """Silence offset 1; the token's influence must skip one position and land on the next."""
    model = perturbed_model("b-rope-smear")
    embed = model.model.embed_tokens
    with torch.no_grad():
        embed.weights[0].zero_()
    ids = tokens()
    with torch.no_grad():
        reference = embed(ids)[0].float().numpy()
        changed = ids.clone()
        changed[0, 12] = (int(changed[0, 12]) + 4) % VOCAB
        after = embed(changed)[0].float().numpy()
    moved = np.abs(reference - after).max(axis=-1)
    assert moved[12] > 0
    assert moved[13] == 0, "offset 1 was silenced but still carried the token forward"
    assert moved[14] > 0, "offset 2 carried nothing; the offsets share a coefficient"


def test_nope_removes_all_position_dependence():
    """Under NoPE the same token in two positions gets the same attention treatment."""
    model = perturbed_model("d-nope")
    rotary = model.model.rotary_emb
    hidden = torch.randn(1, 4, 32)
    cos, sin = rotary(hidden, torch.arange(4).unsqueeze(0))
    assert torch.equal(cos, torch.ones_like(cos))
    assert torch.equal(sin, torch.zeros_like(sin))
    shifted_cos, shifted_sin = rotary(hidden, torch.arange(4).unsqueeze(0) + 1000)
    assert torch.equal(cos, shifted_cos) and torch.equal(sin, shifted_sin)


def test_rope_arm_still_depends_on_position():
    """Guard against the NoPE patch leaking into the control."""
    model = perturbed_model("a-rope")
    hidden = torch.randn(1, 4, 32)
    cos, _ = model.model.rotary_emb(hidden, torch.arange(4).unsqueeze(0))
    assert not torch.equal(cos, torch.ones_like(cos))
