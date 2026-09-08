# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the issue #262 architecture variants.

The causality test is the one that matters. A width-3 smear that leaks even one
position forward turns next-token prediction partly into copying, which shows up
as a beautiful loss curve attached to a worthless model — and nothing else in
the pipeline would catch it.
"""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

from architecture import NoRotaryEmbeddingsConfig, SmearQwen3Config

VOCAB = Axis("vocab", 37)
POSITION = 24


def make_config(**overrides) -> SmearQwen3Config:
    base = dict(
        max_seq_len=POSITION,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        rope=Llama3RotaryEmbeddingsConfig(),
    )
    base.update(overrides)
    return SmearQwen3Config(**base)


def make_model(config: SmearQwen3Config, seed: int = 0):
    return config.model_type.init(VOCAB, config, key=jax.random.PRNGKey(seed))


def random_tokens(seed: int = 1):
    Pos = Axis("position", POSITION)
    values = jax.random.randint(jax.random.PRNGKey(seed), (POSITION,), 0, VOCAB.size)
    return hax.named(values, (Pos,))


def logits(model, tokens):
    from levanter.layers.attention import AttentionMask

    return model(tokens, attn_mask=AttentionMask.causal())


def perturb_smear_weights(model, seed: int = 7):
    """Move the smear off its zero initialisation so the tests can see it."""
    embeddings = model.embeddings
    shape = embeddings.weights.array.shape
    noise = jax.random.normal(jax.random.PRNGKey(seed), shape) * 0.5
    weights = hax.named(jnp.asarray(noise), embeddings.weights.axes)
    return eqx.tree_at(lambda m: m.embeddings.weights, model, weights)


def test_smear_is_strictly_causal():
    """Changing token t+1 must not move the logits at any position <= t.

    This is the whole ballgame: a smear that reaches forward instead of backward
    still trains, still converges, and is worthless.
    """
    model = perturb_smear_weights(make_model(make_config(smear_width=2)))
    tokens = random_tokens()
    reference = logits(model, tokens)

    for cut in (5, 11, POSITION - 2):
        changed = tokens.array.at[cut + 1].set((int(tokens.array[cut + 1]) + 3) % VOCAB.size)
        perturbed = logits(model, hax.named(changed, tokens.axes))
        before = np.asarray(reference.array[: cut + 1])
        after = np.asarray(perturbed.array[: cut + 1])
        assert np.array_equal(before, after), (
            f"changing token {cut + 1} moved the logits at positions <= {cut}: the smear "
            f"is not causal (max abs diff {np.abs(before - after).max()})"
        )


def test_smear_does_not_wrap_around():
    """The first tokens must not see the last ones.

    ``hax.roll`` alone is a circular shift, so this is the specific way the
    causality test above would fail if the mask were dropped.
    """
    model = perturb_smear_weights(make_model(make_config(smear_width=2)))
    tokens = random_tokens()
    reference = logits(model, tokens)

    tail = tokens.array.at[POSITION - 1].set((int(tokens.array[POSITION - 1]) + 5) % VOCAB.size)
    perturbed = logits(model, hax.named(tail, tokens.axes))
    assert np.array_equal(
        np.asarray(reference.array[:2]), np.asarray(perturbed.array[:2])
    ), "the last token reached positions 0-1: the roll is wrapping"


def test_zero_initialised_smear_matches_the_baseline():
    """At initialisation a smear arm is exactly its control, so the arms start together."""
    tokens = random_tokens()
    with_smear = logits(make_model(make_config(smear_width=2)), tokens)
    without = logits(make_model(make_config(smear_width=0)), tokens)
    np.testing.assert_allclose(
        np.asarray(with_smear.array), np.asarray(without.array), rtol=0, atol=0
    )


def test_offsets_have_independent_weights():
    """Offsets 1 and 2 must be separately addressable, or arg1 and arg2 collapse.

    Silence offset 1 while leaving offset 2 alive, then perturb one token. Its
    influence must skip the very next position and land two positions on. A
    coefficient shared across the offsets fails this in both directions at once:
    with w_1 silenced the shared weight is silenced too, so nothing reaches
    ``cut + 2`` either.
    """
    model = perturb_smear_weights(make_model(make_config(smear_width=2)))
    weights = model.embeddings.weights
    Offset = weights.resolve_axis("smear_offset")
    silenced = hax.where(hax.arange(Offset) == 0, 0.0, weights)
    model = eqx.tree_at(lambda m: m.embeddings.weights, model, silenced)

    tokens = random_tokens()
    reference = model.embeddings.embed(tokens)
    cut = 12
    changed = tokens.array.at[cut].set((int(tokens.array[cut]) + 4) % VOCAB.size)
    perturbed = model.embeddings.embed(hax.named(changed, tokens.axes))
    moved = np.abs(np.asarray(reference.array) - np.asarray(perturbed.array)).max(axis=-1)

    assert moved[cut] > 0, "the token's own embedding did not move"
    assert moved[cut + 1] == 0, (
        "offset 1 was silenced but still carried the token forward one position; "
        "the offsets are not independently addressable"
    )
    assert moved[cut + 2] > 0, (
        "offset 2 carried nothing while offset 1 was silenced; the two offsets are "
        "sharing a coefficient"
    )


def test_smear_reaches_exactly_two_tokens_back():
    """Width 2 sees t-1 and t-2 and nothing further.

    Checked at the embedding, not the logits: attention mixes everything
    downstream, so only the embedding can show the smear's true reach.
    """
    model = perturb_smear_weights(make_model(make_config(smear_width=2)))
    tokens = random_tokens()
    reference = model.embeddings.embed(tokens)

    cut = 12
    changed = tokens.array.at[cut].set((int(tokens.array[cut]) + 4) % VOCAB.size)
    perturbed = model.embeddings.embed(hax.named(changed, tokens.axes))
    moved = np.abs(np.asarray(reference.array) - np.asarray(perturbed.array)).max(axis=-1)
    assert moved[cut] > 0 and moved[cut + 1] > 0 and moved[cut + 2] > 0
    assert moved[cut + 3] == 0, "the smear reached three tokens forward; width is wrong"
    assert moved[cut - 1] == 0, "the smear reached backward; it should only push forward"


def test_nope_ignores_position():
    """With NoPE, two attention inputs differing only in position are identical."""
    config = make_config(smear_width=0, rope=NoRotaryEmbeddingsConfig())
    model = make_model(config)
    rotary = config.rope.build(Axis("head_size", 8))
    query = hax.random.normal(jax.random.PRNGKey(3), (Axis("position", 4), Axis("head_size", 8)))
    positions = hax.named(jnp.arange(4), ("position",))
    shifted = hax.named(jnp.arange(4) + 1000, ("position",))
    np.testing.assert_array_equal(
        np.asarray(rotary(query, positions).array), np.asarray(query.array)
    )
    np.testing.assert_array_equal(
        np.asarray(rotary(query, shifted).array), np.asarray(rotary(query, positions).array)
    )
    assert not config.uses_rope


def test_nope_refuses_to_pretend_to_be_hf():
    """A NoPE model has no HF Qwen3 representation and must say so, not guess."""
    with pytest.raises(NotImplementedError, match="cannot be written as an HF Qwen3 config"):
        NoRotaryEmbeddingsConfig().to_hf_config()


def test_parameter_count_includes_the_smear():
    """The smear's cost is two Embed vectors plus a tiny gate — and it is tiny."""
    without = make_config(smear_width=0).total_trainable_params(VOCAB.size)
    with_smear = make_config(smear_width=2).total_trainable_params(VOCAB.size)
    gate_in = min(16, 32)
    assert with_smear - without == 2 * (32 + gate_in + 1)


def test_smear_width_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        make_config(smear_width=-1)
