# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from any_permissible import (
    ContactOracleExample,
    any_permissible_contact_loss,
    contact_edge_capacity,
)
from haliax import Axis
from levanter.layers.attention_mask import AttentionMask


class _FixedLogitModel:
    def __init__(
        self,
        *,
        Pos: Axis,
        Embed: Axis,
        Vocab: Axis,
        activations: hax.NamedArray,
        lm_head: hax.NamedArray,
    ):
        self.Pos = Pos
        self.Embed = Embed
        self.Vocab = Vocab
        self._activations = activations
        self._lm_head = lm_head

    def activations(self, tokens, attn_mask, *, key=None):
        del tokens, attn_mask, key
        return self._activations

    def get_lm_head(self):
        return self._lm_head


def test_contact_loss_matches_explicit_dense_target_distributions():
    Batch = Axis("batch", 1)
    Pos = Axis("position", 12)
    Embed = Axis("embed", 6)
    Vocab = Axis("vocab", 16)
    tokens = np.asarray(
        ((0, 9, 2, 5, 9, 2, 7, 8, 10, 11, 12, 13),),
        dtype=np.int32,
    )
    loss_weight = np.ones(tokens.shape, dtype=np.float32)
    loss_weight[:, -1] = 0
    activations = np.arange(
        Batch.size * Pos.size * Embed.size, dtype=np.float32
    ).reshape(Batch.size, Pos.size, Embed.size)
    activations = (activations - activations.mean()) / 30
    lm_head = np.arange(Embed.size * Vocab.size, dtype=np.float32).reshape(
        Embed.size, Vocab.size
    )
    lm_head = (lm_head - lm_head.mean()) / 40

    edge_capacity = contact_edge_capacity(Pos.size)
    edge_positions = np.zeros((Batch.size, edge_capacity), dtype=np.int32)
    edge_positions[0, :2] = (1, 4)
    edge_segments = np.full((Batch.size, edge_capacity), -1, dtype=np.int32)
    edge_segments[0, :2] = 0
    edge_valid = np.zeros((Batch.size, edge_capacity), dtype=np.bool_)
    edge_valid[0, :2] = True
    segment_ids = hax.zeros((Batch, Pos), dtype=jnp.int32)
    example = ContactOracleExample(
        tokens=hax.named(jnp.asarray(tokens), (Batch, Pos)),
        loss_weight=hax.named(jnp.asarray(loss_weight), (Batch, Pos)),
        attn_mask=AttentionMask.causal().with_segment_ids(segment_ids),
        edge_positions=jnp.asarray(edge_positions),
        edge_segment_ids=jnp.asarray(edge_segments),
        edge_valid=jnp.asarray(edge_valid),
    )
    model = _FixedLogitModel(
        Pos=Pos,
        Embed=Embed,
        Vocab=Vocab,
        activations=hax.named(jnp.asarray(activations), (Batch, Pos, Embed)),
        lm_head=hax.named(jnp.asarray(lm_head), (Embed, Vocab)),
    )

    actual = any_permissible_contact_loss(model, example)

    log_probs = jax.nn.log_softmax(jnp.asarray(activations @ lm_head), axis=-1)
    target_distributions = np.zeros(
        (Batch.size, Pos.size, Vocab.size), dtype=np.float32
    )
    for position in range(Pos.size - 1):
        target_distributions[0, position, tokens[0, position + 1]] = 1
    target_distributions[0, 1] = 0
    target_distributions[0, 1, (2, 5, 7)] = (0.5, 0.25, 0.25)
    target_distributions[0, 2] = 0
    target_distributions[0, 2, (5, 7)] = 0.5
    target_distributions[0, 4] = 0
    target_distributions[0, 4, (2, 7)] = 0.5
    expected = -jnp.sum(
        log_probs
        * jnp.asarray(target_distributions)
        * jnp.asarray(loss_weight)[..., None]
    ) / jnp.sum(jnp.asarray(loss_weight))

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
