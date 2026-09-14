# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent dense CE/gradient oracle, including stock one-hot reduction."""

from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_document
from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config
from test_targets import serialization_oracle

from experiments.exp279_models_exact_soft_contact_targets.model import (
    ContactQwen3Config,
    contact_loss_terms,
)
from experiments.exp279_models_exact_soft_contact_targets.targets import (
    ContactExample,
    DocumentSlice,
    pack_targets,
)


def example_and_oracle(vocab, size=64, edges=((143, 144), (143, 145), (144, 145))):
    doc = make_document(edges)
    tokens = np.pad(doc, (0, size - len(doc)))
    Pos = hax.Axis("position", size)
    base = LmExample.causal(hax.named(tokens, Pos), ignore_id=0)
    target = pack_targets([DocumentSlice(doc)], tokens, vocab, edge_capacity=8)
    example = ContactExample(
        base.tokens, base.loss_weight, base.attn_mask, targets=target
    )
    q = np.eye(vocab.size, dtype=np.float32)[np.roll(tokens, -1)]
    oracle, _ = serialization_oracle(edges)
    structure = int(np.flatnonzero(doc == 9)[0])
    emitted = tuple(doc[structure + 1 : -1])
    for i in range(len(emitted)):
        counts = oracle[emitted[:i]]
        q[structure + i] = 0
        for token, count in counts.items():
            q[structure + i, token] = count / sum(counts.values())
    return example, q


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("block_size", [7, 32, 64])
@pytest.mark.parametrize("scale", [0.2, 8.0])
def test_values_and_both_gradients_against_dense_oracle(
    vocab, dtype, block_size, scale
):
    example, q = example_and_oracle(vocab)
    rng = np.random.default_rng(713)
    x = jnp.asarray(rng.normal(size=(64, 32)) * scale, dtype)
    w = jnp.asarray(rng.normal(size=(32, vocab.size)) * scale, dtype)
    Pos, Embed, Vocab = (
        hax.Axis("position", 64),
        hax.Axis("embed", 32),
        hax.Axis("vocab", vocab.size),
    )
    # Fractional masks and a zero-scored endpoint exercise the actual reduction.
    mask = np.asarray(example.loss_weight.array).copy()
    mask[::3] *= 0.25
    mask[20] = 0
    example = replace(example, loss_weight=hax.named(mask, Pos))

    def actual(x, w):
        ce, _ = contact_loss_terms(
            hax.named(x, (Pos, Embed)),
            hax.named(w, (Embed, Vocab)),
            example,
            block_size=block_size,
            logsumexp_weight=1e-4,
        )
        return hax.nn.loss.reduce_loss(
            ce, hax.mean, None, weight=example.loss_weight
        ).array

    def reference(x, w):
        logits = jnp.matmul(
            x.astype(jnp.float32),
            w.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        ce = (
            -(q * jax.nn.log_softmax(logits)).sum(-1)
            + 1e-4 * jax.nn.logsumexp(logits, axis=-1) ** 2
        )
        return (ce * mask).sum() / mask.sum()

    actual_value, actual_grad = jax.jit(jax.value_and_grad(actual, argnums=(0, 1)))(
        x, w
    )
    expected_value, expected_grad = jax.jit(
        jax.value_and_grad(reference, argnums=(0, 1))
    )(x, w)
    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-6, atol=2e-6)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        # BF16 output gradients have one final rounding. Compare in FP32.
        np.testing.assert_allclose(
            a.astype(jnp.float32),
            b.astype(jnp.float32),
            rtol=0.008 if dtype == jnp.bfloat16 else 2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize("reduction", [None, hax.mean, hax.sum])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_one_hot_is_stock_ce_including_masks(vocab, reduction, dtype):
    example, _ = example_and_oracle(vocab, edges=())
    Pos, Embed, Vocab = (
        hax.Axis("position", 64),
        hax.Axis("embed", 32),
        hax.Axis("vocab", vocab.size),
    )
    x = hax.random.normal(jax.random.PRNGKey(1), (Pos, Embed)).astype(dtype)
    w = hax.random.normal(jax.random.PRNGKey(2), (Embed, Vocab)).astype(dtype)

    def actual(w):
        ce, entropy = contact_loss_terms(x, w, example, block_size=7)
        return hax.nn.loss.reduce_loss(ce, reduction, None, weight=example.loss_weight)

    def stock(w):
        return maybe_fused_next_token_loss(
            Pos,
            Embed,
            Vocab,
            x,
            w,
            example.tokens,
            loss_weight=example.loss_weight,
            reduction=reduction,
        )

    np.testing.assert_allclose(actual(w).array, stock(w).array, rtol=2e-6, atol=2e-6)
    if reduction is not None:
        ga = eqx.filter_grad(lambda w: actual(w).array)(w)
        gb = eqx.filter_grad(lambda w: stock(w).array)(w)
        np.testing.assert_allclose(
            ga.array.astype(jnp.float32),
            gb.array.astype(jnp.float32),
            rtol=0.008 if dtype == jnp.bfloat16 else 3e-5,
            atol=2e-6,
        )


def test_entropy_and_all_masked_loss(vocab):
    example, q = example_and_oracle(vocab)
    Pos, Embed, Vocab = (
        hax.Axis("position", 64),
        hax.Axis("embed", 8),
        hax.Axis("vocab", vocab.size),
    )
    ce, entropy = contact_loss_terms(
        hax.zeros((Pos, Embed)), hax.zeros((Embed, Vocab)), example
    )
    expected_entropy = -(q * np.log(np.maximum(q, 1e-30))).sum(-1)
    np.testing.assert_allclose(entropy.array, expected_entropy, atol=1e-7)
    np.testing.assert_allclose(ce.array, np.log(vocab.size), atol=1e-6)
    value = hax.nn.loss.reduce_loss(ce, hax.mean, None, weight=hax.zeros(Pos))
    assert float(value.array) == 0


def test_stock_model_weights_forward_positions_and_validation(vocab):
    example, _ = example_and_oracle(vocab)
    cfg = ContactQwen3Config(
        max_seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        use_qk_norm=True,
        attn_backend=AttentionBackend.VANILLA,
    )
    key = jax.random.PRNGKey(9)
    model = cfg.build(hax.Axis("vocab", vocab.size), key=key)
    control_cfg = Qwen3Config(
        max_seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        use_qk_norm=True,
        attn_backend=AttentionBackend.VANILLA,
    )
    control = control_cfg.build(hax.Axis("vocab", vocab.size), key=key)
    for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(control), strict=True):
        np.testing.assert_array_equal(a, b)
    a = model(example.tokens, example.attn_mask)
    b = control(example.tokens, example.attn_mask)
    np.testing.assert_array_equal(a.array, b.array)
    ordinary = LmExample(example.tokens, example.loss_weight, example.attn_mask)
    np.testing.assert_array_equal(
        model.compute_next_token_loss(ordinary).array,
        control.compute_next_token_loss(ordinary).array,
    )
    # A future token change must not affect an earlier activation/logit.
    changed = replace(example.tokens, array=example.tokens.array.at[25].set(150))
    altered = model(changed, example.attn_mask)
    np.testing.assert_array_equal(a.array[:25], altered.array[:25])
    assert np.isfinite(float(model.compute_next_token_loss(example).array))
