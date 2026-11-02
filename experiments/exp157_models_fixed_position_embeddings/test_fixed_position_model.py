"""Smoke tests for fixed residue-position input embeddings."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import haliax as hax
from haliax import Axis
from levanter.models.lm_model import LmExample

from fixed_position_model import (
    FixedResiduePositionLlamaConfig,
    ResiduePositionEmbeddingSpec,
    fixed_rope_position_vectors,
)


def _tiny_config() -> FixedResiduePositionLlamaConfig:
    return FixedResiduePositionLlamaConfig(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        tie_word_embeddings=False,
        position_embedding=ResiduePositionEmbeddingSpec(
            start_token_id=10,
            num_tokens=4,
            base=10_000.0,
        ),
    )


def test_fixed_position_embeddings_are_deterministic_and_no_trainable_rows_exist() -> None:
    config = _tiny_config()
    Vocab = Axis("vocab", 20)
    model = config.build(Vocab, key=jax.random.PRNGKey(0))

    assert model.embeddings.Vocab.size == 20
    assert model.embeddings.token_embeddings.Vocab.size == 16

    Pos = Axis("position", 8)
    ids = hax.named(jnp.array([0, 10, 11, 12, 13, 14, 19, 3], dtype=jnp.int32), Pos)
    embedded_before = model.embeddings.embed(ids)

    expected_p0 = fixed_rope_position_vectors(hax.named(jnp.array([0], dtype=jnp.float32), Axis("one", 1)), config.Embed)
    expected_p2 = fixed_rope_position_vectors(hax.named(jnp.array([2], dtype=jnp.float32), Axis("one", 1)), config.Embed)

    assert jnp.allclose(embedded_before.take(Pos, 1).array, expected_p0.array[0])
    assert jnp.allclose(embedded_before.take(Pos, 3).array, expected_p2.array[0])


def test_learned_delta_position_embeddings_start_at_rope_prior() -> None:
    config = dataclasses.replace(
        _tiny_config(),
        position_embedding=ResiduePositionEmbeddingSpec(
            start_token_id=10,
            num_tokens=4,
            base=10_000.0,
            trainable_delta=True,
        ),
    )
    Vocab = Axis("vocab", 20)
    model = config.build(Vocab, key=jax.random.PRNGKey(0))

    assert model.embeddings.token_embeddings.Vocab.size == 20
    position_delta_rows = model.embeddings.token_embeddings.weight.array[10:14]
    assert jnp.allclose(position_delta_rows, 0.0)

    Pos = Axis("position", 4)
    ids = hax.named(jnp.array([10, 11, 12, 13], dtype=jnp.int32), Pos)
    embedded = model.embeddings.embed(ids)
    expected = fixed_rope_position_vectors(hax.named(jnp.arange(4, dtype=jnp.float32), Pos), config.Embed)
    assert jnp.allclose(embedded.array, expected.array)


def test_update_steps_change_model_but_not_fixed_position_embeddings() -> None:
    config = _tiny_config()
    Vocab = Axis("vocab", 20)
    Pos = Axis("position", 8)
    model = config.build(Vocab, key=jax.random.PRNGKey(0))

    # Includes fixed residue-position tokens 10..13 both as context and targets.
    tokens = hax.named(jnp.array([0, 10, 4, 11, 5, 12, 6, 13], dtype=jnp.int32), Pos)
    example = LmExample.causal(tokens)

    fixed_probe = hax.named(jnp.array([0, 1, 2, 3], dtype=jnp.float32), Axis("probe_pos", 4))
    fixed_before = fixed_rope_position_vectors(fixed_probe, config.Embed).array
    learned_before = model.embeddings.token_embeddings.weight.array.copy()
    lm_head_before = model.get_lm_head().array.copy()

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(m):
        return m.compute_next_token_loss(example).array

    for _ in range(3):
        loss, grads = loss_fn(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        assert jnp.isfinite(loss)

    fixed_after = fixed_rope_position_vectors(fixed_probe, config.Embed).array
    learned_delta = jnp.linalg.norm(model.embeddings.token_embeddings.weight.array - learned_before)
    lm_head_delta = jnp.linalg.norm(model.get_lm_head().array - lm_head_before)

    assert jnp.allclose(fixed_after, fixed_before)
    assert learned_delta > 0
    assert lm_head_delta > 0


def test_delta_l2_prior_only_applies_to_reduced_training_loss() -> None:
    config = dataclasses.replace(
        _tiny_config(),
        position_embedding=ResiduePositionEmbeddingSpec(
            start_token_id=10,
            num_tokens=4,
            base=10_000.0,
            trainable_delta=True,
            delta_l2_weight=0.5,
        ),
    )
    Vocab = Axis("vocab", 20)
    Pos = Axis("position", 8)
    model = config.build(Vocab, key=jax.random.PRNGKey(0))
    weight = model.embeddings.token_embeddings.weight.array.at[10:14].set(2.0)
    model = dataclasses.replace(
        model,
        embeddings=dataclasses.replace(
            model.embeddings,
            token_embeddings=dataclasses.replace(
                model.embeddings.token_embeddings,
                weight=hax.named(weight, model.embeddings.token_embeddings.weight.axes),
            ),
        ),
    )

    tokens = hax.named(jnp.array([0, 10, 4, 11, 5, 12, 6, 13], dtype=jnp.int32), Pos)
    example = LmExample.causal(tokens)
    unregularized = dataclasses.replace(
        model,
        embeddings=dataclasses.replace(
            model.embeddings,
            position_spec=dataclasses.replace(model.embeddings.position_spec, delta_l2_weight=0.0),
        ),
    )

    reduced_loss = model.compute_next_token_loss(example).array
    reduced_base = unregularized.compute_next_token_loss(example).array
    assert jnp.allclose(reduced_loss - reduced_base, 2.0)

    unreduced_loss = model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array
    unreduced_base = unregularized.compute_next_token_loss(example, reduction=None, reduction_axis=()).array
    assert jnp.allclose(unreduced_loss, unreduced_base)


def test_update_steps_train_learned_position_delta_embeddings() -> None:
    config = dataclasses.replace(
        _tiny_config(),
        position_embedding=ResiduePositionEmbeddingSpec(
            start_token_id=10,
            num_tokens=4,
            base=10_000.0,
            trainable_delta=True,
        ),
    )
    Vocab = Axis("vocab", 20)
    Pos = Axis("position", 8)
    model = config.build(Vocab, key=jax.random.PRNGKey(0))

    tokens = hax.named(jnp.array([0, 10, 4, 11, 5, 12, 6, 13], dtype=jnp.int32), Pos)
    example = LmExample.causal(tokens)
    position_delta_before = model.embeddings.token_embeddings.weight.array[10:14].copy()

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(m):
        return m.compute_next_token_loss(example).array

    for _ in range(3):
        loss, grads = loss_fn(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        assert jnp.isfinite(loss)

    position_delta_after = model.embeddings.token_embeddings.weight.array[10:14]
    assert jnp.linalg.norm(position_delta_after - position_delta_before) > 0
