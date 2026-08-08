# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Order-marginalized (soft) next-token targets for contacts-v1, on device.

A contacts-v1 document serialises two unordered sets in a uniformly random
order, so the one-hot next-token target is a *sample* from a distribution we can
write down exactly. Training against the distribution instead is a
Rao-Blackwellization: same population objective, lower-variance target. Phase 0
of `#201 <https://github.com/Open-Athena/MarinFold/issues/201>`_ measured that
distribution's entropy at **2.09 nats/token, 77 % of the reported val loss**.

v1 covers the two slot kinds worth 94 % of that:

``statement head``
    Target uniform over the sequence statements not yet emitted (1.13
    nats/token, 54 % of the nuisance floor).
``contact first endpoint``
    Target ``deg_R(p) / (2|R|)`` over the contacts not yet emitted (0.84
    nats/token, 40 %).

Second endpoints (0.12 nats/token, 5.7 %) stay **one-hot**. Their target is
conditioned on the token *at* the slot, which makes it a segmented reverse
cumsum keyed by token value rather than a plain one — real work for a twentieth
of the effect. The resulting loss is still unbiased, just partially
Rao-Blackwellized.

The formulation that makes this cheap
-------------------------------------

Both supported kinds have the same shape: at slot ``i`` the target is a running
count over the *contributing slots at or after* ``i`` **within the same
document**, each contributing the token it predicts::

    statement head slot -> contributes its own next token (a statement head)
    endpoint slot       -> contributes its own next token (a contact endpoint)

and the normaliser is just how many contributed. Written naively that is a
``[position, vocab]`` array. But the loss only ever needs ``<q, z>``, and

    <q, z>_i = (1/norm_i) * sum_p w_i(p) * (h_i . W[p])
             = (1/norm_i) * h_i . (sum_p w_i(p) W[p])

so we can contract with the vocabulary **before** accumulating. ``sum_p w_i(p)
W[p]`` is then a reverse cumulative sum of single embedding rows, and the whole
construction collapses to ``[position, embed]`` — one activation-sized tensor,
no vocabulary axis, no matmul. That is what this module builds.

Packing and document boundaries
-------------------------------

Documents are packed prefix-only into one window, so every accumulation has to
stop at its document's edge. The reverse cumsum ``U`` is taken over the whole
window and then corrected by subtracting the value at a per-slot boundary:

* statement-head slots subtract at the next ``<begin_statements>`` — which also
  excludes this document's own structure section, whose endpoint contributions
  would otherwise leak into the statement target;
* first-endpoint slots subtract at the next ``<begin_sequence>``, i.e. the start
  of the next document (no statement-head contributions occur after
  ``<begin_statements>`` within a document, so nothing else needs excluding).

Both boundaries come from a reverse running minimum over position indices, the
mirror of the running maximum ``loss_masks`` uses for section membership.
"""

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray

# contacts-v1 token ids at the published tokenizer. Defaults only; callers should
# pass ids resolved from the tokenizer they train with (see ``verify_mask.py``).
CONTACT_ID = 5
BEGIN_SEQUENCE_ID = 8
BEGIN_STRUCTURE_ID = 9
END_ID = 10

# Sentinel for "no such position ahead"; indexes the zero row appended to U.
_NO_BOUNDARY = jnp.iinfo(jnp.int32).max


def _cummax(values: NamedArray, axis: Axis) -> NamedArray:
    return hax.named(
        jax.lax.cummax(values.array, axis=values.axes.index(axis)), values.axes
    )


def _reverse_cummin(values: NamedArray, axis: Axis) -> NamedArray:
    return hax.named(
        jax.lax.cummin(values.array, axis=values.axes.index(axis), reverse=True),
        values.axes,
    )


def _reverse_cumsum(values: NamedArray, axis: Axis) -> NamedArray:
    return hax.named(
        jax.lax.cumsum(values.array, axis=values.axes.index(axis), reverse=True),
        values.axes,
    )


def slot_kinds(
    tokens: NamedArray,
    *,
    contact_id: int = CONTACT_ID,
    begin_sequence_id: int = BEGIN_SEQUENCE_ID,
    begin_structure_id: int = BEGIN_STRUCTURE_ID,
    end_id: int = END_ID,
) -> dict[str, NamedArray]:
    """Classify every slot of a (possibly packed) token window.

    Slot ``i`` predicts ``tokens[i + 1]``, matching ``LmExample.loss_weight``.

    Args:
        tokens: Token ids, ``[..., position]``.
        contact_id: Id of ``<contact>``.
        begin_sequence_id: Id of ``<begin_sequence>``.
        begin_structure_id: Id of ``<begin_statements>``.
        end_id: Id of ``<end>``.

    Returns:
        Boolean masks keyed ``statement_head``, ``first_endpoint``,
        ``second_endpoint`` and ``endpoint`` (first or second — the slots that
        *contribute* to a first-endpoint target).
    """
    Pos = tokens.axes[-1]
    index = hax.arange(Pos)
    next_token = hax.roll(tokens, -1, Pos)
    previous_token = hax.roll(tokens, 1, Pos)
    # The window's last slot has no next token; roll wrapped it around.
    has_next = index < (Pos.size - 1)

    is_closer = (tokens == begin_structure_id) | (tokens == end_id)
    last_begin_sequence = _cummax(hax.where(tokens == begin_sequence_id, index, -1), Pos)
    last_closer = _cummax(hax.where(is_closer, index, -1), Pos)
    in_sequence_section = last_begin_sequence > last_closer

    next_is_closer = (next_token == begin_structure_id) | (next_token == end_id)
    statement_head = (
        in_sequence_section
        & (((index - last_begin_sequence) % 2) == 0)
        & ~next_is_closer
        & has_next
    )

    last_begin_structure = _cummax(hax.where(tokens == begin_structure_id, index, -1), Pos)
    last_end = _cummax(hax.where(tokens == end_id, index, -1), Pos)
    in_structure_section = last_begin_structure > last_end

    first_endpoint = in_structure_section & (tokens == contact_id) & has_next
    second_endpoint = in_structure_section & (previous_token == contact_id) & has_next
    return {
        "statement_head": statement_head,
        "first_endpoint": first_endpoint,
        "second_endpoint": second_endpoint,
        "endpoint": first_endpoint | second_endpoint,
    }


def _boundary_index(tokens: NamedArray, marker_id: int, Pos: Axis) -> NamedArray:
    """For each slot, the index of the next ``marker_id`` at or after it."""
    index = hax.arange(Pos)
    marked = hax.where(tokens == marker_id, index, _NO_BOUNDARY)
    return _reverse_cummin(marked, Pos)


def _accumulate(
    contribution: NamedArray, boundary: NamedArray, Pos: Axis
) -> NamedArray:
    """Reverse cumulative sum of ``contribution``, truncated at ``boundary``.

    ``contribution`` is ``[..., position, embed]`` (or ``[..., position]`` for the
    scalar normaliser) and ``boundary`` is ``[..., position]``. The running total
    from a slot to the end of the window is corrected by subtracting the total
    from ``boundary`` onward, which is what stops a document accumulating over its
    neighbours in a packed window.

    Done on the raw arrays: the gather indexes the position axis with an index
    that itself carries a position axis, which is not something haliax's named
    ``take`` expresses cleanly.
    """
    axis = contribution.axes.index(Pos)
    running = jax.lax.cumsum(contribution.array, axis=axis, reverse=True)
    # A zero slice past the end, so an out-of-range boundary subtracts nothing.
    padding = [(0, 0)] * running.ndim
    padding[axis] = (0, 1)
    padded = jnp.pad(running, padding)

    index = jnp.clip(boundary.array, 0, Pos.size)
    while index.ndim < running.ndim:
        index = index[..., None]
    index = jnp.broadcast_to(index, running.shape)
    tail = jnp.take_along_axis(padded, index, axis=axis)
    return hax.named(running - tail, contribution.axes)


def soft_target_directions(
    tokens: NamedArray,
    lm_head: NamedArray,
    *,
    Vocab: Axis,
    Embed: Axis,
    contact_id: int = CONTACT_ID,
    begin_sequence_id: int = BEGIN_SEQUENCE_ID,
    begin_structure_id: int = BEGIN_STRUCTURE_ID,
    end_id: int = END_ID,
) -> tuple[NamedArray, NamedArray, NamedArray]:
    """Vocabulary-contracted soft targets for a (possibly packed) token window.

    Args:
        tokens: Token ids, ``[..., position]``.
        lm_head: The output embedding, ``[vocab, embed]``.
        Vocab: The vocabulary axis of ``lm_head``.
        Embed: The embedding axis of ``lm_head``.
        contact_id: Id of ``<contact>``.
        begin_sequence_id: Id of ``<begin_sequence>``.
        begin_structure_id: Id of ``<begin_statements>``.
        end_id: Id of ``<end>``.

    Returns:
        ``(direction, normalizer, is_soft)``. At a soft slot,
        ``dot(h_i, direction_i) / normalizer_i`` equals ``<q_i, z_i>`` — the term
        the soft cross-entropy needs. ``direction`` is ``[..., position, embed]``,
        the other two ``[..., position]``. At a hard slot ``is_soft`` is False and
        the other two are meaningless.
    """
    Pos = tokens.axes[-1]
    kinds = slot_kinds(
        tokens,
        contact_id=contact_id,
        begin_sequence_id=begin_sequence_id,
        begin_structure_id=begin_structure_id,
        end_id=end_id,
    )
    next_token = hax.roll(tokens, -1, Pos)
    # The embedding row of the token each slot predicts, zeroed where the slot
    # does not contribute to the kind being accumulated.
    # [..., position, embed] -- the axis order _accumulate's gather assumes.
    predicted_row = lm_head.take(Vocab, next_token).rearrange((..., Pos, Embed))

    def accumulate(contributes: NamedArray, marker_id: int):
        gate = contributes.astype(predicted_row.dtype)
        boundary = _boundary_index(tokens, marker_id, Pos)
        direction = _accumulate(predicted_row * gate, boundary, Pos)
        normalizer = _accumulate(contributes.astype(jnp.float32), boundary, Pos)
        return direction, normalizer

    # Statement heads stop at <begin_statements>: that boundary excludes both the
    # next document AND this document's own structure section.
    head_direction, head_normalizer = accumulate(
        kinds["statement_head"], begin_structure_id
    )
    # First endpoints accumulate over BOTH endpoints of every remaining contact
    # (that is the factor 2 in deg_R(p) / 2|R|), stopping at the next document.
    endpoint_direction, endpoint_normalizer = accumulate(
        kinds["endpoint"], begin_sequence_id
    )

    is_head = kinds["statement_head"]
    is_first = kinds["first_endpoint"]
    is_soft = is_head | is_first
    direction = hax.where(is_head, head_direction, endpoint_direction)
    normalizer = hax.where(is_head, head_normalizer, endpoint_normalizer)
    return direction, normalizer, is_soft


__all__ = [
    "BEGIN_SEQUENCE_ID",
    "BEGIN_STRUCTURE_ID",
    "CONTACT_ID",
    "END_ID",
    "slot_kinds",
    "soft_target_directions",
]
