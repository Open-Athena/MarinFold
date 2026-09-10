import jax
import jax.numpy as jnp
import numpy as np

from marinfold_models.document_loss import _sparse_contact_example_losses_from_arrays


def _reference_sparse_contact_example_losses(
    activations_array,
    log_normalizers_array,
    lm_head_by_vocab,
    contact_first_ids,
    contact_second_ids,
    second_neighbor_ids,
    second_neighbor_counts,
    second_neighbor_count,
    contact_count,
    prediction_start,
    *,
    contact_token_id: int,
    end_token_id: int,
):
    def one_example_loss(
        activations_one,
        log_z_one,
        first_ids,
        second_ids,
        neighbor_ids,
        neighbor_counts,
        neighbor_count,
        example_contact_count,
        example_prediction_start,
    ):
        valid_contacts = jnp.arange(first_ids.shape[0], dtype=jnp.int32) < example_contact_count
        endpoint_rows = lm_head_by_vocab[first_ids] + lm_head_by_vocab[second_ids]
        endpoint_sum0 = jnp.sum(jnp.where(valid_contacts[:, None], endpoint_rows, 0.0), axis=0)

        def logit(position, token_id):
            position = jnp.clip(position, 0, activations_one.shape[0] - 1)
            return jnp.sum(activations_one[position] * lm_head_by_vocab[token_id], axis=-1)

        def cross_entropy(position, expected_logit):
            position = jnp.clip(position, 0, log_z_one.shape[0] - 1)
            return log_z_one[position] - expected_logit

        def body(c, carry):
            endpoint_sum, total = carry
            valid = c < example_contact_count
            contact_position = example_prediction_start + 1 + 3 * c
            first_position = contact_position + 1
            contact_predict_position = jnp.where(c == 0, example_prediction_start, contact_position - 1)

            contact_position = jnp.clip(contact_position, 0, activations_one.shape[0] - 1)
            first_position = jnp.clip(first_position, 0, activations_one.shape[0] - 1)
            contact_loss = cross_entropy(contact_predict_position, logit(contact_predict_position, contact_token_id))
            first_expected_logit = jnp.sum(activations_one[contact_position] * endpoint_sum) / jnp.maximum(
                2 * (example_contact_count - c), 1
            )
            first_loss = cross_entropy(contact_position, first_expected_logit)

            neighbor_rows = lm_head_by_vocab[neighbor_ids[c]]
            neighbor_logits = jnp.sum(activations_one[first_position] * neighbor_rows, axis=-1)
            second_expected_logit = jnp.sum(neighbor_counts[c] * neighbor_logits) / jnp.maximum(
                neighbor_count[c], 1
            )
            second_loss = cross_entropy(first_position, second_expected_logit)

            current_endpoint_rows = lm_head_by_vocab[first_ids[c]] + lm_head_by_vocab[second_ids[c]]
            next_endpoint_sum = endpoint_sum - jnp.where(valid, current_endpoint_rows, 0.0)
            next_total = total + jnp.where(valid, contact_loss + first_loss + second_loss, 0.0)
            return next_endpoint_sum, next_total

        _, body_loss = jax.lax.fori_loop(0, first_ids.shape[0], body, (endpoint_sum0, jnp.asarray(0.0, jnp.float32)))
        end_position = jnp.clip(example_prediction_start + 3 * example_contact_count, 0, log_z_one.shape[0] - 1)
        end_loss = cross_entropy(end_position, logit(end_position, end_token_id))
        return body_loss + end_loss

    return jax.vmap(one_example_loss)(
        activations_array,
        log_normalizers_array,
        contact_first_ids,
        contact_second_ids,
        second_neighbor_ids,
        second_neighbor_counts,
        second_neighbor_count,
        contact_count,
        prediction_start,
    )


def _random_sparse_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    batch = 4
    pos = 18
    embed = 7
    vocab = 23
    max_contacts = 5
    max_degree = 4

    activations = jnp.asarray(rng.normal(size=(batch, pos, embed)).astype(np.float32))
    log_z = jnp.asarray(rng.normal(size=(batch, pos)).astype(np.float32))
    lm_head = jnp.asarray(rng.normal(size=(vocab, embed)).astype(np.float32))
    contact_count = jnp.asarray([0, 1, 3, 5], dtype=jnp.int32)
    prediction_start = jnp.asarray([1, 2, 0, 1], dtype=jnp.int32)

    first_ids_np = rng.integers(2, vocab, size=(batch, max_contacts), dtype=np.int32)
    second_ids_np = rng.integers(2, vocab, size=(batch, max_contacts), dtype=np.int32)
    neighbor_ids_np = rng.integers(2, vocab, size=(batch, max_contacts, max_degree), dtype=np.int32)
    neighbor_counts_np = rng.integers(0, 4, size=(batch, max_contacts, max_degree)).astype(np.float32)
    neighbor_count_np = neighbor_counts_np.sum(axis=-1).astype(np.int32)

    for row, count in enumerate(np.asarray(contact_count)):
        first_ids_np[row, count:] = 0
        second_ids_np[row, count:] = 0
        neighbor_ids_np[row, count:, :] = 0
        neighbor_counts_np[row, count:, :] = 0
        neighbor_count_np[row, count:] = 0

    return (
        activations,
        log_z,
        lm_head,
        jnp.asarray(first_ids_np),
        jnp.asarray(second_ids_np),
        jnp.asarray(neighbor_ids_np),
        jnp.asarray(neighbor_counts_np),
        jnp.asarray(neighbor_count_np),
        contact_count,
        prediction_start,
    )


def test_vectorized_sparse_contact_loss_matches_running_sum_reference():
    inputs = _random_sparse_inputs()
    actual = _sparse_contact_example_losses_from_arrays(*inputs, contact_token_id=0, end_token_id=1)
    expected = _reference_sparse_contact_example_losses(*inputs, contact_token_id=0, end_token_id=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_vectorized_sparse_contact_loss_gradients_match_reference():
    inputs = _random_sparse_inputs(seed=1)

    def actual_loss(activations, lm_head):
        updated = (activations, inputs[1], lm_head, *inputs[3:])
        return jnp.sum(_sparse_contact_example_losses_from_arrays(*updated, contact_token_id=0, end_token_id=1))

    def expected_loss(activations, lm_head):
        updated = (activations, inputs[1], lm_head, *inputs[3:])
        return jnp.sum(_reference_sparse_contact_example_losses(*updated, contact_token_id=0, end_token_id=1))

    actual_grads = jax.jit(jax.grad(actual_loss, argnums=(0, 1)))(inputs[0], inputs[2])
    expected_grads = jax.jit(jax.grad(expected_loss, argnums=(0, 1)))(inputs[0], inputs[2])

    for actual, expected in zip(actual_grads, expected_grads):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
