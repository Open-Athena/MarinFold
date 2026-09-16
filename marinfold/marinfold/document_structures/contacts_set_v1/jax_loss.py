# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX loss for contacts-set-v1 target-anchored slot assignment.

The loss treats the 16 contact slots on each residue as an unordered set.  Each
present target contact is assigned to one unique predicted slot.  Predicted slots
not used by the best assignment receive the empty-slot penalty.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .format import COARSE_BINS_PER_SIGN, CONTACT_SLOTS, FINE_BINS

NUM_SIGNED_COARSE_BINS = 2 * COARSE_BINS_PER_SIGN
NUM_FINE_BINS = FINE_BINS
_NUM_MASKS = 1 << CONTACT_SLOTS
_MASKS = jnp.arange(_NUM_MASKS, dtype=jnp.uint32)
_SLOT_BITS = (jnp.uint32(1) << jnp.arange(CONTACT_SLOTS, dtype=jnp.uint32))
_INF = jnp.asarray(1.0e30, dtype=jnp.float32)


@dataclass(frozen=True)
class ContactSetLossWeights:
    """Weights for the decomposed contacts-set-v1 slot loss."""

    present: float = 1.0
    coarse: float = 1.0
    fine: float = 1.0
    extra: float = 1.0


class ContactSetLoss(NamedTuple):
    """Scalar loss and assignment diagnostics for one residue or document."""

    total: jax.Array
    assigned: jax.Array
    extra: jax.Array
    present_positive: jax.Array
    coarse: jax.Array
    fine: jax.Array
    present_negative: jax.Array
    best_mask: jax.Array
    num_targets: jax.Array


def sigmoid_binary_cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Elementwise numerically stable BCE-with-logits."""
    targets = targets.astype(logits.dtype)
    return jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def softmax_cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Elementwise sparse softmax cross entropy over the final logits axis."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_indices = jnp.broadcast_to(targets, log_probs.shape[:-1])[..., None]
    return -jnp.take_along_axis(log_probs, target_indices, axis=-1)[..., 0]


def _extra_empty_loss_by_mask(empty_loss_by_slot: jax.Array) -> jax.Array:
    slot_is_used = (_MASKS[:, None] & _SLOT_BITS[None, :]) != 0
    return jnp.sum(jnp.where(slot_is_used, 0.0, empty_loss_by_slot[None, :]), axis=1)


def _target_assignment_cost_by_slot(
    present_logits: jax.Array,
    signed_coarse_logits: jax.Array,
    fine_logits: jax.Array,
    target_signed_coarse: jax.Array,
    target_fine: jax.Array,
    weights: ContactSetLossWeights,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    positive_present = sigmoid_binary_cross_entropy(present_logits, jnp.ones_like(present_logits))
    coarse = softmax_cross_entropy(signed_coarse_logits, target_signed_coarse)
    fine = softmax_cross_entropy(fine_logits, target_fine)
    total = weights.present * positive_present + weights.coarse * coarse + weights.fine * fine
    return total, positive_present, coarse, fine


def contacts_set_residue_loss(
    present_logits: jax.Array,
    signed_coarse_logits: jax.Array,
    fine_logits: jax.Array,
    target_present: jax.Array,
    target_signed_coarse: jax.Array,
    target_fine: jax.Array,
    weights: ContactSetLossWeights = ContactSetLossWeights(),
) -> ContactSetLoss:
    """Compute the permutation-invariant contacts-set-v1 loss for one residue.

    Args:
        present_logits: Float array with shape ``[16]``.
        signed_coarse_logits: Float array with shape ``[16, 256]``.
        fine_logits: Float array with shape ``[16, 16]``.
        target_present: Boolean array with shape ``[16]``. Present targets are
            interpreted as the target contact set; their storage order does not
            affect the minimum assignment cost.
        target_signed_coarse: Integer array with shape ``[16]``.
        target_fine: Integer array with shape ``[16]``.
        weights: Decomposed loss weights.

    Returns:
        Total loss and diagnostics for the best target-anchored slot assignment.
    """
    present_logits = jnp.asarray(present_logits)
    signed_coarse_logits = jnp.asarray(signed_coarse_logits)
    fine_logits = jnp.asarray(fine_logits)
    target_present = jnp.asarray(target_present, dtype=jnp.bool_)
    target_signed_coarse = jnp.asarray(target_signed_coarse, dtype=jnp.int32)
    target_fine = jnp.asarray(target_fine, dtype=jnp.int32)

    present_negative_by_slot = sigmoid_binary_cross_entropy(
        present_logits, jnp.zeros_like(present_logits, dtype=jnp.bool_)
    )
    extra_by_mask = weights.extra * _extra_empty_loss_by_mask(present_negative_by_slot)
    present_negative_by_mask = _extra_empty_loss_by_mask(present_negative_by_slot)

    initial_total = jnp.full((_NUM_MASKS,), _INF, dtype=present_logits.dtype).at[0].set(0.0)
    initial_component = jnp.zeros((_NUM_MASKS,), dtype=present_logits.dtype)
    initial_state = (initial_total, initial_component, initial_component, initial_component)

    def assign_one(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        target: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        total_states, present_states, coarse_states, fine_states = state
        is_present, coarse_target, fine_target = target
        cost_by_slot, present_by_slot, coarse_by_slot, fine_by_slot = _target_assignment_cost_by_slot(
            present_logits,
            signed_coarse_logits,
            fine_logits,
            coarse_target,
            fine_target,
            weights,
        )

        # For each resulting mask M, the candidates are exactly the slots s in
        # M, with previous mask M \ {s}.  Reducing over those <=16 candidates
        # gives the same target-anchored DP as the reference implementation and
        # lets us carry component sums from the winning predecessor.
        mask_has_slot = (_MASKS[:, None] & _SLOT_BITS[None, :]) != 0
        prev_masks = (_MASKS[:, None] & ~_SLOT_BITS[None, :]).astype(jnp.int32)
        candidate_totals = total_states[prev_masks] + cost_by_slot[None, :]
        candidate_totals = jnp.where(mask_has_slot, candidate_totals, _INF)
        best_slot = jnp.argmin(candidate_totals, axis=1)
        next_totals = jnp.min(candidate_totals, axis=1)
        best_prev_mask = prev_masks[jnp.arange(_NUM_MASKS), best_slot]
        next_present = present_states[best_prev_mask] + present_by_slot[best_slot]
        next_coarse = coarse_states[best_prev_mask] + coarse_by_slot[best_slot]
        next_fine = fine_states[best_prev_mask] + fine_by_slot[best_slot]
        next_state = (next_totals, next_present, next_coarse, next_fine)
        return tuple(jnp.where(is_present, new, old) for new, old in zip(next_state, state))

    assigned_by_mask, present_positive_by_mask, coarse_by_mask, fine_by_mask = jax.lax.scan(
        lambda state, target: (assign_one(state, target), None),
        initial_state,
        (target_present, target_signed_coarse, target_fine),
    )[0]
    total_by_mask = assigned_by_mask + extra_by_mask
    best_mask = jnp.argmin(total_by_mask)
    assigned = assigned_by_mask[best_mask]
    extra = extra_by_mask[best_mask]
    total = assigned + extra
    return ContactSetLoss(
        total=total,
        assigned=assigned,
        extra=extra,
        present_positive=present_positive_by_mask[best_mask],
        coarse=coarse_by_mask[best_mask],
        fine=fine_by_mask[best_mask],
        present_negative=present_negative_by_mask[best_mask],
        best_mask=best_mask.astype(jnp.uint32),
        num_targets=jnp.sum(target_present.astype(jnp.int32)),
    )


def contacts_set_document_loss(
    present_logits: jax.Array,
    signed_coarse_logits: jax.Array,
    fine_logits: jax.Array,
    target_present: jax.Array,
    target_signed_coarse: jax.Array,
    target_fine: jax.Array,
    weights: ContactSetLossWeights = ContactSetLossWeights(),
) -> ContactSetLoss:
    """Sum contacts-set-v1 loss over a residue sequence.

    Arrays have a leading residue axis followed by the same per-residue shapes as
    :func:`contacts_set_residue_loss`.
    """
    losses = jax.vmap(
        lambda pl, scl, fl, tp, tsc, tf: contacts_set_residue_loss(pl, scl, fl, tp, tsc, tf, weights)
    )(present_logits, signed_coarse_logits, fine_logits, target_present, target_signed_coarse, target_fine)
    return ContactSetLoss(
        total=jnp.sum(losses.total),
        assigned=jnp.sum(losses.assigned),
        extra=jnp.sum(losses.extra),
        present_positive=jnp.sum(losses.present_positive),
        coarse=jnp.sum(losses.coarse),
        fine=jnp.sum(losses.fine),
        present_negative=jnp.sum(losses.present_negative),
        best_mask=losses.best_mask,
        num_targets=jnp.sum(losses.num_targets),
    )
