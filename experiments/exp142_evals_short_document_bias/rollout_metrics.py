# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse a contacts-v1 rollout into a predicted contact set (for scoring) AND
into an *ordered* list of contacts with per-contact logprobs (for the exp102
accuracy-factor analysis, issue #102).

``parse_pred`` / ``score_rollout`` / ``gt_by_band`` are copied verbatim from
exp98 so per-band precision/recall/F1 stay identical and the new
``rollout_metrics_ordered.parquet`` joins 1:1 to exp98's ``rollout_metrics_all``.

The exp98 worker discarded emission order (it stored ``sorted(pred)``) and kept
only a whole-rollout ``nll``. exp102 additionally keeps, per rollout:

  * the predicted contacts in **generation order** (first occurrence wins on
    dedup), and
  * each contact's **logprob at emission** (sum of the 3 sampled-token logprobs
    of its ``<contact> <pI> <pJ>`` statement).

Both come from ``parse_contacts_ordered``, which walks the *token* stream rather
than regex-matching decoded text: in the contacts-v1 tokenizer every position
token ``<pN>`` and the ``<contact>`` marker is a single vocab id, so a contact
statement is exactly the token triple ``(<contact>, <pI>, <pJ>)``. Walking tokens
lets us line each statement up with its per-token logprobs exactly.

Pure Python (no torch / marinfold) so it imports anywhere and is unit-testable.
"""
from __future__ import annotations

import math
import re

CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
POS_RE = re.compile(r"^<p(\d+)>$")
CONTACT_TOK = "<contact>"
MIN_SEP = 6
# (lo, hi) inclusive separation bands; hi=None == unbounded.
BANDS: dict[str, tuple[int, int | None]] = {
    "all": (MIN_SEP, None),
    "short": (6, 11),
    "med": (12, 23),
    "long": (24, None),
}


def parse_pred(text: str, pos_to_seq: dict[int, int]) -> set[tuple[int, int]]:
    """Predicted contact set (seq-index pairs, deduped, seq-sep >= MIN_SEP).

    Verbatim from exp98 — the scoring path. Kept so exp102 metrics match exp98.
    """
    out: set[tuple[int, int]] = set()
    for a, b in CONTACT_RE.findall(text):
        ia, ib = pos_to_seq.get(int(a)), pos_to_seq.get(int(b))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        out.add((min(ia, ib), max(ia, ib)))
    return out


def parse_contacts_ordered(
    token_strs: list[str], pos_to_seq: dict[int, int]
) -> list[tuple[int, int, int]]:
    """Contacts in **emission order**, deduped (first occurrence wins).

    ``token_strs`` are the generated tokens as strings (e.g. ``"<contact>"``,
    ``"<p996>"``), from ``tokenizer.convert_ids_to_tokens(gen_ids)``. Returns a
    list of ``(i, j, k)`` where ``(i, j)`` is the seq-index contact (``i < j``,
    same dedup/sep>=6 filter as ``parse_pred``) and ``k`` is the index of the
    statement's ``<contact>`` token in ``token_strs`` — so the caller can read
    the per-token logprobs at ``k, k+1, k+2`` for the contact's emission logprob.

    A malformed statement (``<contact>`` not followed by two ``<pN>`` tokens, or
    a position not in ``pos_to_seq``, or sep < MIN_SEP, or i == j) is skipped,
    exactly mirroring ``parse_pred``'s filters. This makes the ordered list a
    permutation of ``parse_pred(text)`` (order + repeats aside).
    """
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int, int]] = []
    n = len(token_strs)
    for k in range(n):
        if token_strs[k] != CONTACT_TOK:
            continue
        if k + 2 >= n:
            break
        ma = POS_RE.match(token_strs[k + 1])
        mb = POS_RE.match(token_strs[k + 2])
        if not (ma and mb):
            continue
        ia = pos_to_seq.get(int(ma.group(1)))
        ib = pos_to_seq.get(int(mb.group(1)))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        pair = (min(ia, ib), max(ia, ib))
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append((pair[0], pair[1], k))
    return ordered


def _in_band(pair: tuple[int, int], lo: int, hi: int | None) -> bool:
    s = abs(pair[0] - pair[1])
    return s >= lo and (hi is None or s <= hi)


def gt_by_band(gt: set[tuple[int, int]]) -> dict[str, set[tuple[int, int]]]:
    return {name: {g for g in gt if _in_band(g, lo, hi)} for name, (lo, hi) in BANDS.items()}


def score_rollout(pred: set[tuple[int, int]],
                  gtb: dict[str, set[tuple[int, int]]]) -> dict[str, float]:
    """Flat per-band metrics dict: ``{band}_npred/_tp/_prec/_rec/_f1``.

    Verbatim from exp98. recall is NaN for a band with no ground-truth contacts.
    """
    row: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        P = {p for p in pred if _in_band(p, lo, hi)}
        G = gtb[name]
        tp = len(P & G)
        prec = tp / len(P) if P else 0.0
        rec = tp / len(G) if G else math.nan
        if math.isnan(rec):
            f1 = math.nan
        elif prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        row[f"{name}_npred"] = len(P)
        row[f"{name}_tp"] = tp
        row[f"{name}_prec"] = prec
        row[f"{name}_rec"] = rec
        row[f"{name}_f1"] = f1
    return row
