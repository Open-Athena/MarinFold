# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp163 loss-weight profiles over a contacts-v1 document — numpy only.

Two document formats live here.

**v1 — "candidate" format** (Phases 1–3, ``answer_span_loss_weights``). Candidate
rollouts were prepended in blocks marked with a repurposed spare token, and the
loss was armed *only* on the single ``<begin_statements> … <end>`` answer span::

    <contacts-v1> <begin_sequence> …sequence…      weight 0
      <CAND> <contact> pi pj …                     weight 0
    <begin_statements> …TRUE contacts… <end>       weight 1

**v2 — "multi-draft" format** (``span_weights``). The ``<CAND>`` marker is gone:
``<begin_statements>`` now means *"discard what came before; here is a new
candidate structure"*, and **only the final section is closed by ``<end>``**::

    <contacts-v1> <begin_sequence> …sequence…      w_header
    <begin_statements> …draft 1…                   w_draft
    <begin_statements> …draft 2…                   w_draft   (conditions on draft 1)
    <begin_statements> …TRUE contacts… <end>       w_final

Drafts are ordered by ascending quality, so the document is a refinement ramp and
the LAST section is the answer.

Why v2 exists: it makes a rollout and its own drafts the *same* object, so a
single generation can emit N successive structures and be RL'd on best-of-N (or
on the last) without any candidate/output distinction. The cost is that the
model no longer gets a syntactic "this is untrusted" signal and must infer draft
vs answer from position.

Both tokens keep their existing meanings, so ``<end>`` remains the generation
stop token and no inference path changes.

WEIGHT CONVENTION. levanter computes ``target_y = roll(true_ids, -1)``, so
``weight[i]`` weights the loss on predicting ``tokens[i+1]``. Sections are
half-open ``[start, stop)``, which puts the branch decisions in the right place:

* ``weight[start]``  -> predicts the section's first ``<contact>``
* for a draft, ``weight[stop-1]`` -> predicts the next ``<begin_statements>``:
  the **restart** decision, at ``w_draft``
* for the final section, ``weight[stop-1]`` -> predicts ``<end>``: the **stop**
  decision, at ``w_final``
* ``weight[stop]`` for the final section (the ``<end>`` position itself, which
  predicts the packing ``<eos>``) falls through to ``w_header`` — matching v1,
  where the packed ``<eos>`` was never trained.

So with ``(0, 0, 1)`` on a single-section document the profile is **identical** to
v1's ``answer_span_loss_weights``.
"""

from __future__ import annotations

import numpy as np

# Ids in the contacts-v1 tokenizer are fully determined by
# ``marinfold.document_structures.core.build_tokenizer``: it prepends ``<pad>``=0,
# ``<eos>``=1, then ``all_domain_tokens()`` at id 2+. These defaults are a
# convenience for callers without a tokenizer handy; the corpus tokenizer always
# resolves them from the live tokenizer via :func:`resolve_span_ids`.
BEGIN_STATEMENTS_ID = 9  # <begin_statements>          (BEGIN_STRUCTURE_TOKEN)
END_ID = 10              # <end>                       (END_TOKEN)
CANDIDATE_MARKER_ID = 7  # <contacts-and-distances-v1> (v1's <CAND> marker only)

BEGIN_STATEMENTS_TOKEN = "<begin_statements>"
END_TOKEN = "<end>"
CANDIDATE_MARKER_TOKEN = "<contacts-and-distances-v1>"


def resolve_span_ids(tokenizer) -> tuple[int, int]:
    """Resolve ``(<begin_statements>, <end>)`` ids from a live contacts-v1 tokenizer.

    Raises if either token is unknown, and if the resolved ids drift from the
    baked-in constants — those are documented in several places and a silent
    drift would weight the wrong tokens.
    """
    begin = tokenizer.convert_tokens_to_ids(BEGIN_STATEMENTS_TOKEN)
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    unk = getattr(tokenizer, "unk_token_id", None)
    for token, tid in ((BEGIN_STATEMENTS_TOKEN, begin), (END_TOKEN, end)):
        if tid is None or (unk is not None and tid == unk):
            raise ValueError(f"tokenizer does not know {token!r} — is this the contacts-v1 tokenizer?")
    if (begin, end) != (BEGIN_STATEMENTS_ID, END_ID):
        raise ValueError(
            f"contacts-v1 vocab drift: resolved <begin_statements>/<end> ids ({begin}, {end}) "
            f"!= baked-in ({BEGIN_STATEMENTS_ID}, {END_ID}). Update loss_mask.py's constants "
            f"(and re-check every place that documents them) before building a corpus."
        )
    return begin, end


def find_sections(
    ids: np.ndarray,
    *,
    begin_id: int = BEGIN_STATEMENTS_ID,
    end_id: int = END_ID,
) -> list[tuple[int, int]]:
    """``[(start, stop_exclusive), …]`` for every statements section, in order.

    v2 grammar: a new section begins at each ``<begin_statements>``, and **only
    the final section is closed by ``<end>``**. Drafts are simply superseded by
    the next ``<begin_statements>``.

    That keeps both tokens' existing meanings — ``<end>`` still terminates the
    document, so it remains the generation stop token and no inference path has
    to change. Termination becomes a learned choice: after a contact triple the
    model emits another ``<contact>``, or ``<begin_statements>`` to start over,
    or ``<end>`` to finish.

    Ranges are half-open so ``ids[start:stop]`` is the section's own tokens:
    draft *k* runs up to the next ``<begin_statements>``, and the final section
    runs up to (not including) ``<end>``.
    """
    ids = np.asarray(ids)
    begins = [int(i) for i in np.flatnonzero(ids == begin_id)]
    ends = [int(i) for i in np.flatnonzero(ids == end_id)]
    if not begins:
        raise ValueError("document has no <begin_statements> section")
    if len(ends) != 1:
        raise ValueError(f"expected exactly one <end> (document terminator), got {len(ends)}")
    stop = ends[0]
    if stop < begins[-1]:
        raise ValueError(f"<end> at {stop} precedes the last <begin_statements> at {begins[-1]}")
    bounds = begins[1:] + [stop]
    return list(zip(begins, bounds))


def span_weights(
    ids: np.ndarray,
    *,
    w_header: float,
    w_draft: float,
    w_final: float,
    begin_id: int = BEGIN_STATEMENTS_ID,
    end_id: int = END_ID,
) -> np.ndarray:
    """Per-position float32 loss weight for ONE multi-draft document.

    The last statements section is the answer (``w_final``); any earlier ones are
    drafts (``w_draft``); everything else — the sequence header, and the ``<end>``
    position itself — is ``w_header``. Sections are the half-open ranges
    :func:`find_sections` returns; see the module docstring for why that puts the
    restart / stop decisions at the right positions.

    This is deliberately a per-document function, not a packed-sequence one: the
    corpus tokenizer computes it per document and concatenates, so "which section
    is last" is unambiguous. (v1's ``answer_span_loss_weights`` was packing-safe
    by construction because it only needed a binary indicator; a *ramp* needs to
    know where each document ends.)
    """
    ids = np.asarray(ids)
    w = np.full(len(ids), float(w_header), dtype=np.float32)
    sections = find_sections(ids, begin_id=begin_id, end_id=end_id)
    for k, (start, stop) in enumerate(sections):
        w[start:stop] = w_final if k == len(sections) - 1 else w_draft
    return w


def answer_span_loss_weights(
    ids: np.ndarray,
    *,
    begin_id: int = BEGIN_STATEMENTS_ID,
    end_id: int = END_ID,
) -> np.ndarray:
    """v1 mask: 1.0 on each answer span, 0.0 elsewhere (packing-safe).

    Kept because the Phase 1–3 corpus was built with it and the results in
    ``WRITEUP.md`` refer to it. Equivalent to
    ``span_weights(w_header=0, w_draft=0, w_final=1)`` on a single-section
    document, but implemented as two running counts so it also works on an
    already-packed sequence:

        opened[i] = # of <begin_statements> in ids[:i+1]   (inclusive cumsum)
        closed[i] = # of <end>              in ids[:i+1]
        weight[i] = 1.0 if opened[i] > closed[i] else 0.0

    Note this arms ``[b, e)`` — it excludes ``weight[e]``, so the token after
    ``<end>`` is never trained. That was right when ``<end>`` terminated the
    document; it is exactly what :func:`span_weights` changes.
    """
    ids = np.asarray(ids)
    opened = np.cumsum(ids == begin_id)
    closed = np.cumsum(ids == end_id)
    return (opened > closed).astype(np.float32)


def check_document_spans(
    ids: np.ndarray,
    *,
    begin_id: int = BEGIN_STATEMENTS_ID,
    end_id: int = END_ID,
) -> tuple[int, int]:
    """v1 invariant: assert exactly ONE well-formed section; return its bounds."""
    sections = find_sections(ids, begin_id=begin_id, end_id=end_id)
    if len(sections) != 1:
        raise ValueError(f"expected exactly one statements section, got {len(sections)}")
    return sections[0]


__all__ = [
    "BEGIN_STATEMENTS_ID",
    "CANDIDATE_MARKER_ID",
    "END_ID",
    "answer_span_loss_weights",
    "check_document_spans",
    "find_sections",
    "resolve_span_ids",
    "span_weights",
]
