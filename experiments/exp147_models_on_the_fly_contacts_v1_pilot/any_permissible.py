# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contact-specific dynamic-oracle loss for the exp147 causal documents."""

from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis
from levanter.data.dataset import AsyncDataset, MappedAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import named_attention_mask_from_grug
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    fused_cross_entropy_loss_and_logsumexp_penalty,
)
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.schedule import BatchSchedule

CONTACT_EDGE_CHUNK_SIZE = 256


@dataclass(frozen=True)
class GrugContactOracleExample:
    """Unnamed causal LM example plus its ordered contact-edge slots."""

    tokens: jax.Array
    loss_weight: jax.Array
    attn_mask: object
    edge_positions: jax.Array
    edge_segment_ids: jax.Array
    edge_valid: jax.Array


class ContactOracleExample(eqx.Module):
    """Named causal LM example plus its ordered contact-edge slots."""

    tokens: hax.NamedArray
    loss_weight: hax.NamedArray
    attn_mask: object
    edge_positions: jax.Array
    edge_segment_ids: jax.Array
    edge_valid: jax.Array


def contact_edge_capacity(max_seq_len: int) -> int:
    """Return a chunk-aligned upper bound on contact statements in a pack."""
    max_edges = max_seq_len // 3
    return (
        (max_edges + CONTACT_EDGE_CHUNK_SIZE - 1) // CONTACT_EDGE_CHUNK_SIZE
    ) * CONTACT_EDGE_CHUNK_SIZE


class NamedContactOracleDataset(
    MappedAsyncDataset[GrugContactOracleExample, ContactOracleExample]
):
    """Name token axes without discarding the contact-oracle metadata."""

    def __init__(self, dataset: AsyncDataset[GrugContactOracleExample], Pos: Axis):
        self.dataset = dataset
        self.Pos = Pos

        def _to_named(example: GrugContactOracleExample) -> ContactOracleExample:
            return ContactOracleExample(
                tokens=hax.named(example.tokens, Pos),
                loss_weight=hax.named(example.loss_weight, Pos),
                attn_mask=named_attention_mask_from_grug(example.attn_mask, Pos),
                edge_positions=example.edge_positions,
                edge_segment_ids=example.edge_segment_ids,
                edge_valid=example.edge_valid,
            )

        super().__init__(dataset, _to_named)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


class ContactOracleDataConfig(LmDataConfig):
    """Use the contact-aware naming adapter for the sole train component."""

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: jax.Array,
    ) -> AsyncDataset[ContactOracleExample]:
        mix_key, shuffle_key = jax.random.split(key)
        datasets = self.train_sets(
            Pos,
            key=shuffle_key,
            initial_batch_size=batch_schedule.batch_size_at_step(0),
        )
        if not isinstance(self.train_weights, dict):
            raise TypeError(
                "The exp147 contact-oracle run requires static dictionary weights"
            )
        mixture = MixtureDataset(
            datasets=datasets,
            weights=self.train_weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )
        return NamedContactOracleDataset(mixture, Pos)


def _selected_logits(
    activations: jax.Array,
    token_ids: jax.Array,
    lm_head: jax.Array,
) -> jax.Array:
    return jnp.sum(activations * lm_head[token_ids], axis=-1)


def _oracle_expected_logits(
    *,
    first_query_activations: jax.Array,
    second_query_activations: jax.Array,
    first_endpoint_ids: jax.Array,
    second_endpoint_ids: jax.Array,
    endpoint_embeddings: jax.Array,
    edge_positions: jax.Array,
    edge_segments: jax.Array,
    edge_valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute exact oracle expectations in bounded-memory edge chunks."""
    batch_size, edge_capacity = edge_positions.shape
    if edge_capacity % CONTACT_EDGE_CHUNK_SIZE:
        raise ValueError(
            f"Edge capacity {edge_capacity} is not divisible by "
            f"{CONTACT_EDGE_CHUNK_SIZE}"
        )

    zeros = jnp.zeros((batch_size, edge_capacity), dtype=jnp.float32)

    def accumulate(chunk_index, totals):
        first_numerator, first_denominator, second_numerator, second_denominator = (
            totals
        )
        start = chunk_index * CONTACT_EDGE_CHUNK_SIZE
        candidate_positions = jax.lax.dynamic_slice_in_dim(
            edge_positions, start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )
        candidate_segments = jax.lax.dynamic_slice_in_dim(
            edge_segments, start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )
        candidate_valid = jax.lax.dynamic_slice_in_dim(
            edge_valid, start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )
        candidate_first_ids = jax.lax.dynamic_slice_in_dim(
            first_endpoint_ids, start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )
        candidate_second_ids = jax.lax.dynamic_slice_in_dim(
            second_endpoint_ids, start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )
        candidate_first_embeddings = jax.lax.dynamic_slice_in_dim(
            endpoint_embeddings[:, 0], start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )
        candidate_second_embeddings = jax.lax.dynamic_slice_in_dim(
            endpoint_embeddings[:, 1], start, CONTACT_EDGE_CHUNK_SIZE, axis=1
        )

        remaining = (
            edge_valid[:, :, None]
            & candidate_valid[:, None, :]
            & (edge_segments[:, :, None] == candidate_segments[:, None, :])
            & (candidate_positions[:, None, :] >= edge_positions[:, :, None])
        )
        remaining_float = remaining.astype(jnp.float32)
        first_to_first = jnp.einsum(
            "beh,bch->bec",
            first_query_activations,
            candidate_first_embeddings,
        )
        first_to_second = jnp.einsum(
            "beh,bch->bec",
            first_query_activations,
            candidate_second_embeddings,
        )
        first_numerator += jnp.sum(
            remaining_float * (first_to_first + first_to_second), axis=-1
        )
        first_denominator += 2.0 * jnp.sum(remaining_float, axis=-1)

        realized_first = first_endpoint_ids[:, :, None]
        matches_first = remaining & (candidate_first_ids[:, None, :] == realized_first)
        matches_second = remaining & (
            candidate_second_ids[:, None, :] == realized_first
        )
        second_to_first = jnp.einsum(
            "beh,bch->bec",
            second_query_activations,
            candidate_first_embeddings,
        )
        second_to_second = jnp.einsum(
            "beh,bch->bec",
            second_query_activations,
            candidate_second_embeddings,
        )
        second_numerator += jnp.sum(
            matches_first.astype(jnp.float32) * second_to_second
            + matches_second.astype(jnp.float32) * second_to_first,
            axis=-1,
        )
        second_denominator += jnp.sum(matches_first, axis=-1) + jnp.sum(
            matches_second, axis=-1
        )
        return (
            first_numerator,
            first_denominator,
            second_numerator,
            second_denominator,
        )

    first_num, first_den, second_num, second_den = jax.lax.fori_loop(
        0,
        edge_capacity // CONTACT_EDGE_CHUNK_SIZE,
        accumulate,
        (zeros, zeros, zeros, zeros),
    )
    return (
        first_num / jnp.maximum(first_den, 1.0),
        second_num / jnp.maximum(second_den, 1.0),
    )


def any_permissible_contact_loss(
    model: Qwen3LMHeadModel,
    example: ContactOracleExample,
    *,
    key=None,
    logsumexp_weight: float | None = None,
) -> jax.Array:
    """Apply ordinary LM loss plus the exact remaining-edge oracle correction.

    The realized document order remains teacher-forced. At each ``<contact>``
    input, either endpoint of every remaining edge is permissible, with mass
    proportional to incidence count. At the realized first endpoint, every
    remaining incident edge contributes its other endpoint. Because
    ``CE(p) = CE(y) + logit(y) - E_p[logit]``, this correction reuses the
    standard fused one-hot cross entropy and never materializes dense soft
    labels over the vocabulary.
    """
    activations = model.activations(
        example.tokens,
        example.attn_mask,
        key=key,
    )
    aux_loss = 0
    if isinstance(activations, tuple):
        activations, aux_loss = activations

    Pos = model.Pos
    standard_loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        activations.rearrange((..., model.Embed)).array.reshape((-1, model.Embed.size)),
        hax.roll(example.tokens, -1, Pos).array.reshape((-1,)),
        model.get_lm_head().rearrange((model.Embed, model.Vocab)).array,
        weight=example.loss_weight.array.reshape((-1,)),
        logsumexp_weight=logsumexp_weight,
        dtype=jnp.float32,
    )

    token_array = example.tokens.array
    activation_array = activations.rearrange(
        (*example.tokens.axes[:-1], Pos, model.Embed)
    ).array
    if token_array.ndim != 2:
        raise ValueError(
            f"Contact-oracle training expects batched tokens, got {token_array.shape}"
        )

    batch_size, _ = token_array.shape
    batch_indices = jnp.arange(batch_size)[:, None]
    edge_positions = example.edge_positions.astype(jnp.int32)
    edge_valid = example.edge_valid.astype(jnp.bool_)
    edge_segments = example.edge_segment_ids.astype(jnp.int32)

    first_query_activations = activation_array[batch_indices, edge_positions]
    second_query_activations = activation_array[batch_indices, edge_positions + 1]
    first_endpoint_ids = token_array[batch_indices, edge_positions + 1]
    second_endpoint_ids = token_array[batch_indices, edge_positions + 2]

    lm_head = model.get_lm_head().rearrange((model.Vocab, model.Embed)).array
    endpoint_embeddings = jnp.stack(
        (lm_head[first_endpoint_ids], lm_head[second_endpoint_ids]), axis=1
    )
    expected_first_logit, expected_second_logit = _oracle_expected_logits(
        first_query_activations=first_query_activations,
        second_query_activations=second_query_activations,
        first_endpoint_ids=first_endpoint_ids,
        second_endpoint_ids=second_endpoint_ids,
        endpoint_embeddings=endpoint_embeddings,
        edge_positions=edge_positions,
        edge_segments=edge_segments,
        edge_valid=edge_valid,
    )

    realized_first_logit = _selected_logits(
        first_query_activations,
        first_endpoint_ids,
        lm_head,
    )
    realized_second_logit = _selected_logits(
        second_query_activations,
        second_endpoint_ids,
        lm_head,
    )
    first_loss_weight = example.loss_weight.array[batch_indices, edge_positions]
    second_loss_weight = example.loss_weight.array[batch_indices, edge_positions + 1]
    correction = edge_valid.astype(jnp.float32) * (
        first_loss_weight * (realized_first_logit - expected_first_logit)
        + second_loss_weight * (realized_second_logit - expected_second_logit)
    )
    loss_weight_sum = jnp.sum(example.loss_weight.array, dtype=jnp.float32)
    corrected_loss = standard_loss + jnp.where(
        loss_weight_sum > 0,
        jnp.sum(correction) / loss_weight_sum,
        0.0,
    )
    return corrected_loss + aux_loss


class AnyPermissibleQwen3LMHeadModel(Qwen3LMHeadModel):
    """Qwen3 model dispatching contact-oracle train examples to the new loss."""

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: Qwen3Config,
        *,
        key,
    ) -> "AnyPermissibleQwen3LMHeadModel":
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(self, example, *, key=None, **kwargs):
        if isinstance(example, ContactOracleExample):
            return any_permissible_contact_loss(
                self,
                example,
                key=key,
                logsumexp_weight=kwargs.get("logsumexp_weight"),
            )
        return LlamaLMHeadModel.compute_next_token_loss(
            self,
            example,
            key=key,
            **kwargs,
        )


@dataclass(frozen=True)
class AnyPermissibleQwen3Config(Qwen3Config):
    """Qwen3 config whose model recognizes contact-oracle train examples."""

    @property
    def model_type(self):
        return AnyPermissibleQwen3LMHeadModel


__all__ = [
    "AnyPermissibleQwen3Config",
    "ContactOracleDataConfig",
    "ContactOracleExample",
    "GrugContactOracleExample",
    "any_permissible_contact_loss",
    "contact_edge_capacity",
]
