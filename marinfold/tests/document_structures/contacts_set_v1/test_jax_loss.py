# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX tests for the contacts-set-v1 target-anchored slot loss."""

import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from marinfold.document_structures.contacts_set_v1.format import CONTACT_SLOTS, encode_delta
from marinfold.document_structures.contacts_set_v1.jax_loss import (
    NUM_FINE_BINS,
    NUM_SIGNED_COARSE_BINS,
    ContactSetLossWeights,
    contacts_set_residue_loss,
    sigmoid_binary_cross_entropy,
    softmax_cross_entropy,
)


def _blank_logits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = np.full((CONTACT_SLOTS,), -4.0, dtype=np.float32)
    coarse = np.zeros((CONTACT_SLOTS, NUM_SIGNED_COARSE_BINS), dtype=np.float32)
    fine = np.zeros((CONTACT_SLOTS, NUM_FINE_BINS), dtype=np.float32)
    return present, coarse, fine


def _target_from_deltas(deltas: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = np.zeros((CONTACT_SLOTS,), dtype=bool)
    coarse = np.zeros((CONTACT_SLOTS,), dtype=np.int32)
    fine = np.zeros((CONTACT_SLOTS,), dtype=np.int32)
    for idx, delta in enumerate(deltas):
        present[idx] = True
        coarse[idx], fine[idx] = encode_delta(delta)
    return present, coarse, fine


def _set_slot_logits(
    present_logits: np.ndarray,
    coarse_logits: np.ndarray,
    fine_logits: np.ndarray,
    *,
    slot: int,
    delta: int,
    present_logit: float = 4.0,
) -> None:
    coarse, fine = encode_delta(delta)
    present_logits[slot] = present_logit
    coarse_logits[slot, coarse] = 6.0
    fine_logits[slot, fine] = 6.0


def _brute_force_loss(
    present_logits: np.ndarray,
    coarse_logits: np.ndarray,
    fine_logits: np.ndarray,
    target_present: np.ndarray,
    target_coarse: np.ndarray,
    target_fine: np.ndarray,
    weights: ContactSetLossWeights = ContactSetLossWeights(),
) -> float:
    targets = np.flatnonzero(target_present)
    empty_by_slot = np.asarray(sigmoid_binary_cross_entropy(jnp.asarray(present_logits), jnp.zeros_like(jnp.asarray(present_logits))))
    best = np.inf
    for slots in itertools.permutations(range(CONTACT_SLOTS), len(targets)):
        used = set(slots)
        assigned = 0.0
        for target_idx, slot in zip(targets, slots):
            assigned += weights.present * float(sigmoid_binary_cross_entropy(jnp.asarray(present_logits[slot]), jnp.asarray(True)))
            assigned += weights.coarse * float(
                softmax_cross_entropy(jnp.asarray(coarse_logits[slot]), jnp.asarray(target_coarse[target_idx]))
            )
            assigned += weights.fine * float(
                softmax_cross_entropy(jnp.asarray(fine_logits[slot]), jnp.asarray(target_fine[target_idx]))
            )
        extra = weights.extra * float(sum(empty_by_slot[slot] for slot in range(CONTACT_SLOTS) if slot not in used))
        best = min(best, assigned + extra)
    if len(targets) == 0:
        best = weights.extra * float(empty_by_slot.sum())
    return best


def test_residue_loss_is_target_order_invariant_and_jittable():
    present_logits, coarse_logits, fine_logits = _blank_logits()
    # Predictions deliberately store the same contact set in different slots from
    # the target order.  The loss should choose the set assignment, not slotwise
    # position 0->0, 1->1.
    _set_slot_logits(present_logits, coarse_logits, fine_logits, slot=7, delta=17)
    _set_slot_logits(present_logits, coarse_logits, fine_logits, slot=2, delta=42)
    target_present, target_coarse, target_fine = _target_from_deltas([42, 17])

    loss = jax.jit(contacts_set_residue_loss)(
        jnp.asarray(present_logits),
        jnp.asarray(coarse_logits),
        jnp.asarray(fine_logits),
        jnp.asarray(target_present),
        jnp.asarray(target_coarse),
        jnp.asarray(target_fine),
    )

    expected = _brute_force_loss(present_logits, coarse_logits, fine_logits, target_present, target_coarse, target_fine)
    assert float(loss.total) == pytest.approx(expected, rel=1e-5)
    assert int(loss.num_targets) == 2
    assert int(loss.best_mask) == (1 << 2) | (1 << 7)


def test_extra_slot_penalty_prefers_using_present_prediction_for_target():
    present_logits, coarse_logits, fine_logits = _blank_logits()
    _set_slot_logits(present_logits, coarse_logits, fine_logits, slot=0, delta=12, present_logit=5.0)
    _set_slot_logits(present_logits, coarse_logits, fine_logits, slot=1, delta=12, present_logit=-5.0)
    target_present, target_coarse, target_fine = _target_from_deltas([12])

    loss = contacts_set_residue_loss(
        jnp.asarray(present_logits),
        jnp.asarray(coarse_logits),
        jnp.asarray(fine_logits),
        jnp.asarray(target_present),
        jnp.asarray(target_coarse),
        jnp.asarray(target_fine),
    )

    assert int(loss.best_mask) == 1 << 0


def test_spot_checked_row_like_contacts_have_one_loss_position_per_residue():
    # First spot-checked row residue 8 had deltas +17, +18, +42, +46.  The JAX
    # loss consumes exactly one [16]-slot target record for that residue.
    deltas = [17, 18, 42, 46]
    present_logits, coarse_logits, fine_logits = _blank_logits()
    for slot, delta in zip([4, 1, 9, 0], deltas):
        _set_slot_logits(present_logits, coarse_logits, fine_logits, slot=slot, delta=delta)
    target_present, target_coarse, target_fine = _target_from_deltas(deltas)

    loss = contacts_set_residue_loss(
        jnp.asarray(present_logits),
        jnp.asarray(coarse_logits),
        jnp.asarray(fine_logits),
        jnp.asarray(target_present),
        jnp.asarray(target_coarse),
        jnp.asarray(target_fine),
    )

    assert target_present.shape == (CONTACT_SLOTS,)
    assert int(loss.num_targets) == 4
    assert int(loss.best_mask) == (1 << 0) | (1 << 1) | (1 << 4) | (1 << 9)
    assert np.isfinite(float(loss.total))
