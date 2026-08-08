# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""On-device loss masks derived from a contacts-v1 token stream.

The contacts-v1 sequence section is a **uniformly random shuffle** of the
``<pX> <AA>`` / ``<n-term> <pX>`` / ``<c-term> <pX>`` statements, so the slot
that predicts each statement's *first* token is asking the model which statement
the generator happened to emit next. Measured over the exp53 validation split
that is **1.13 nats/token — 42 % of the whole contacts-v1 training loss** — and
all of it is nuisance: those slots are prompt, not prediction. See
`#201 <https://github.com/Open-Athena/MarinFold/issues/201>`_.

This module builds the mask that zeroes them, from token ids alone. It needs no
side-channel, no cache change and no soft-target kernel — the masked loss is
levanter's ordinary fused cross-entropy with a modified ``loss_weight``.

Lives here rather than in ``marinfold.document_structures`` because it is JAX;
the plain-Python oracle it is tested against is
``marinfold...contacts_v1.soft_targets.statement_head_slots``.
"""

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray

# contacts-v1 token ids at the published tokenizer (`timodonnell/contacts-v1-tokenizer`,
# 2846 entries; ids 0/1 are <pad>/<eos>). Defaults only — every caller should
# pass ids resolved from the tokenizer it actually trains with, and assert them.
BEGIN_SEQUENCE_ID = 8
BEGIN_STRUCTURE_ID = 9
END_ID = 10


def _cummax(values: NamedArray, axis: hax.Axis) -> NamedArray:
    """Running maximum along ``axis`` (haliax has ``cumsum`` but no ``cummax``)."""
    return hax.named(
        jax.lax.cummax(values.array, axis=values.axes.index(axis)), values.axes
    )


def contacts_v1_statement_head_mask(
    tokens: NamedArray,
    *,
    begin_sequence_id: int = BEGIN_SEQUENCE_ID,
    section_closer_ids: tuple[int, ...] = (BEGIN_STRUCTURE_ID, END_ID),
    dtype: jnp.dtype = jnp.float32,
) -> NamedArray:
    """Multiplicative loss mask: 1.0 to keep a slot, 0.0 on statement-head slots.

    Slot ``i`` predicts ``tokens[i + 1]``, matching ``LmExample.loss_weight``. A
    slot is a statement-head slot when all three hold:

    1. it lies inside a sequence section — the most recent ``<begin_sequence>``
       is more recent than the most recent section closer;
    2. its offset from that ``<begin_sequence>`` is **even** — the section is
       ``<begin_sequence> head body head body ...``, so the ``<begin_sequence>``
       slot itself and every body slot predict a head;
    3. the token it predicts is **not** a section closer. Without this the slot
       that predicts ``<begin_statements>`` would be masked, and that token is
       real information ("the sequence has ended"), not nuisance.

    Everything else keeps full weight — amino acids, terminus indices, section
    markers, and the entire structure section.

    Packing-safe by construction: both "most recent" lookups are running maxima
    over position indices, so a later document's ``<begin_sequence>`` always wins
    over an earlier document's closer without any per-document reset. A window
    that starts partway through a document leaves ``last_begin_sequence`` at -1,
    so nothing is masked rather than something wrong being masked.

    Args:
        tokens: Token ids, ``[..., position]``.
        begin_sequence_id: Id of ``<begin_sequence>``.
        section_closer_ids: Ids that close a sequence section — ``<begin_statements>``
            for a full document, ``<end>`` for the sequence-only variant.
        dtype: Output dtype; multiply straight into ``LmExample.loss_weight``.

    Returns:
        A mask shaped like ``tokens``.
    """
    if not section_closer_ids:
        raise ValueError("section_closer_ids must not be empty")

    Pos = tokens.axes[-1]
    index = hax.arange(Pos)

    is_closer = _is_in(tokens, section_closer_ids)
    last_begin = _cummax(hax.where(tokens == begin_sequence_id, index, -1), Pos)
    last_closer = _cummax(hax.where(is_closer, index, -1), Pos)

    in_sequence_section = last_begin > last_closer
    predicts_head_position = ((index - last_begin) % 2) == 0
    predicts_closer = _is_in(hax.roll(tokens, -1, Pos), section_closer_ids)

    is_statement_head = in_sequence_section & predicts_head_position & ~predicts_closer
    return (~is_statement_head).astype(dtype)


def _is_in(tokens: NamedArray, ids: tuple[int, ...]) -> NamedArray:
    matches = tokens == ids[0]
    for token_id in ids[1:]:
        matches = matches | (tokens == token_id)
    return matches


__all__ = [
    "BEGIN_SEQUENCE_ID",
    "BEGIN_STRUCTURE_ID",
    "END_ID",
    "contacts_v1_statement_head_mask",
]
