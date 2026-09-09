# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Full production loss dimensions, independent dense BF16 value/gradient oracle."""

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_document
from levanter.models.lm_model import LmExample

from experiments.exp279_models_exact_soft_contact_targets.model import (
    contact_loss_terms,
)
from experiments.exp279_models_exact_soft_contact_targets.targets import (
    ContactExample,
    DocumentSlice,
    pack_targets,
)


@pytest.mark.accelerator
def test_full_context_bf16_logits_and_gradients(vocab):
    if jax.default_backend() == "cpu":
        pytest.skip("Run with JAX_PLATFORMS=cuda,cpu and the local-gpu extra")
    edges = [(143, 144 + i) for i in range(1500)]
    doc = make_document(edges)
    tokens = np.pad(doc, (0, 8192 - len(doc)))
    Pos, Embed, Vocab = (
        hax.Axis("position", 8192),
        hax.Axis("embed", 2048),
        hax.Axis("vocab", vocab.size),
    )
    base = LmExample.causal(hax.named(jnp.asarray(tokens), Pos), ignore_id=0)
    targets = pack_targets([DocumentSlice(doc)], tokens, vocab, edge_capacity=2731)
    example = ContactExample(
        base.tokens, base.loss_weight, base.attn_mask, targets=targets
    )
    q = np.eye(vocab.size, dtype=np.float64)[np.roll(tokens, -1)]
    # Independent mutable graph oracle. It explicitly materializes all targets;
    # production stores only ordered edges/ranges and builds one block at a time.
    remaining = set(edges)
    for marker in np.flatnonzero(tokens == 5):
        q[marker : marker + 2] = 0
        for a, b in remaining:
            q[marker, a] += 1 / (2 * len(remaining))
            q[marker, b] += 1 / (2 * len(remaining))
        first, second = tokens[marker + 1 : marker + 3]
        neighbors = [
            b if a == first else a for a, b in remaining if a == first or b == first
        ]
        q[marker + 1, neighbors] = 1 / len(neighbors)
        remaining.remove(tuple(sorted((first, second))))
    q = q.astype(np.float32)
    rng = np.random.default_rng(279)
    x = jnp.asarray(rng.normal(size=(8192, 2048)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(scale=0.2, size=(2048, vocab.size)), jnp.bfloat16)
    mask = base.loss_weight.array

    def actual(x, w):
        ce, _ = contact_loss_terms(
            hax.named(x, (Pos, Embed)), hax.named(w, (Embed, Vocab)), example
        )
        return (ce.array * mask).sum() / mask.sum()

    def reference(x, w):
        logits = jnp.matmul(
            x.astype(jnp.float32),
            w.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        return (
            -(jnp.asarray(q) * jax.nn.log_softmax(logits)).sum(-1) * mask
        ).sum() / mask.sum()

    value, grads = jax.jit(jax.value_and_grad(actual, (0, 1)))(x, w)
    expected, expected_grads = jax.jit(jax.value_and_grad(reference, (0, 1)))(x, w)
    np.testing.assert_allclose(value, expected, atol=1e-5, rtol=1e-6)
    for a, b in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(
            a.astype(jnp.float32), b.astype(jnp.float32), atol=2e-6, rtol=0.008
        )
