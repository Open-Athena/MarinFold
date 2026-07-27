# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The exp163 answer-span loss mask (issue #163) — dependency-light, numpy only.

An exp163 refinement document (built by ``build_refinement_corpus.py``) is::

    <contacts-v1> <begin_sequence> ...sequence...          <- header      (weight 0)
      <CAND> <contact> pi pj ...                           <- candidate 1 (weight 0)
      ...                                                     0..K blocks
      <CAND> <contact> pi pj ...                           <- candidate K (weight 0)
    <begin_statements> <contact> ...TRUE contacts... <end>  <- answer span (weight 1)

The model must learn to PRODUCE the true contacts from the noisy candidate
evidence — never to reproduce the sequence or the candidates. So the LM loss is
armed only on the ``<begin_statements> … <end>`` answer span.

WHERE THIS RUNS. In the executor-era levanter, a ``loss_weight_fn`` could be
hung on the training ``DatasetComponent`` and evaluated on the data-loading
worker. Current levanter (1.2, marin 0.2.57) dropped that field: per-token loss
weights now arrive **from the cache**, via
``PrebuiltLmDatasetFormat(input_ids_key=..., loss_weights_key=...)``. So exp163
computes the mask **offline**, in ``tokenize_refinement_corpus.py``, and ships it
as a ``loss_weights`` column next to ``input_ids``.

That is strictly better than the old arrangement for two reasons:

* nothing has to cloudpickle across the fray boundary any more — the training
  worker reads plain arrays, so it never needs to import exp163 code (which it
  could not do: ``marinfold`` pins ``transformers``/``huggingface_hub<1.0``,
  incompatible with the marin training stack);
* the mask is computed where the real tokenizer is available, so the special-token
  ids are RESOLVED rather than assumed (see :func:`resolve_span_ids`).

Convention (levanter ``models/loss.py``: ``target_y = roll(true_ids, -1)``):
``loss_weight[i]`` weights the loss on predicting ``tokens[i+1]``.
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
CANDIDATE_MARKER_ID = 7  # <contacts-and-distances-v1> (the repurposed <CAND> marker)

BEGIN_STATEMENTS_TOKEN = "<begin_statements>"
END_TOKEN = "<end>"
CANDIDATE_MARKER_TOKEN = "<contacts-and-distances-v1>"


def resolve_span_ids(tokenizer) -> tuple[int, int]:
    """Resolve ``(<begin_statements>, <end>)`` ids from a live contacts-v1 tokenizer.

    Raises if either token is unknown, and warns loudly (via ``ValueError``) if
    the resolved ids drift from the baked-in constants — the constants are
    documented in several places and a silent drift would mask the answer span
    on the wrong tokens.
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


def answer_span_loss_weights(
    ids: np.ndarray,
    *,
    begin_id: int = BEGIN_STATEMENTS_ID,
    end_id: int = END_ID,
) -> np.ndarray:
    """Per-position float32 loss weight: 1.0 on each answer span, 0.0 elsewhere.

    Works on a single document's ids OR on a packed sequence of several
    documents — the indicator is built from two running counts, so it re-arms at
    every ``<begin_statements>`` and disarms at the matching ``<end>``::

        opened[i] = # of <begin_statements> in ids[:i+1]   (inclusive cumsum)
        closed[i] = # of <end>              in ids[:i+1]   (inclusive cumsum)
        weight[i] = 1.0 if opened[i] > closed[i] else 0.0

    Walk one document with ``<begin_statements>`` at index ``b`` and ``<end>`` at
    index ``e`` (``b < e``). The header and every ``<CAND>`` block contain neither
    token (``<CAND>`` is a different id), so neither count moves there:

    ======  ================  ======  ============================================
    pos i   (opened, closed)  weight  weights predicting ids[i+1] = ...
    ======  ================  ======  ============================================
    < b     equal             0.0     header / candidate tokens        -> excluded
    b - 1   equal             0.0     ids[b] = <begin_statements>      -> excluded
    b       closed + 1        1.0     ids[b+1] = FIRST true <contact>  -> TRAINED
    b..e-1  closed + 1        1.0     each <contact> <pX> <pY> triple  -> TRAINED
    e - 1   closed + 1        1.0     ids[e] = <end>: the STOP token   -> TRAINED
    e       equal             0.0     ids[e+1] = <eos> / next doc      -> excluded
    > e     equal             0.0                                      -> excluded
    ======  ================  ======  ============================================

    The inclusive cumsum supplies the ``loss_weight[i] -> ids[i+1]`` shift
    implicitly: counting ``<begin_statements>`` AT index ``b`` arms the mask at
    ``weight[b]``, and counting ``<end>`` AT index ``e`` disarms it at
    ``weight[e]`` while leaving ``weight[e-1]`` armed. No extra roll is needed.
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
    """Assert ONE well-formed ``<begin_statements> … <end>`` span; return its bounds.

    The cumsum mask only stays balanced if each document opens and closes exactly
    once, in that order. A malformed document would silently mis-weight its
    neighbours once packed, so the corpus builder checks every document.
    """
    ids = np.asarray(ids)
    begins = np.flatnonzero(ids == begin_id)
    ends = np.flatnonzero(ids == end_id)
    if len(begins) != 1 or len(ends) != 1:
        raise ValueError(
            f"expected exactly one <begin_statements> and one <end>, got "
            f"{len(begins)} and {len(ends)}"
        )
    b, e = int(begins[0]), int(ends[0])
    if b >= e:
        raise ValueError(f"<begin_statements> at {b} must precede <end> at {e}")
    return b, e


__all__ = [
    "BEGIN_STATEMENTS_ID",
    "CANDIDATE_MARKER_ID",
    "END_ID",
    "answer_span_loss_weights",
    "check_document_spans",
    "resolve_span_ids",
]
