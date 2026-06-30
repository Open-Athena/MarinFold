# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse a contacts-v1 rollout completion into a predicted contact set and score
it (precision / recall / F1) against ground truth, per separation band.

Pure Python (regex + sets) — no marinfold, no torch — so it imports in the marin
TPU worker env and is unit-testable locally without a model. Contact statements
are ``<contact> <pi> <pj>``; positions are mapped back to sequence indices via the
realization's ``pos -> seq_index`` map, then deduped and filtered to seq-sep >= 6
(the contacts-v1 contact definition).
"""
from __future__ import annotations

import math
import re

CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
MIN_SEP = 6
# (lo, hi) inclusive separation bands; hi=None == unbounded.
BANDS: dict[str, tuple[int, int | None]] = {
    "all": (MIN_SEP, None),
    "short": (6, 11),
    "med": (12, 23),
    "long": (24, None),
}


def parse_pred(text: str, pos_to_seq: dict[int, int]) -> set[tuple[int, int]]:
    """Predicted contact set (seq-index pairs, deduped, seq-sep >= MIN_SEP)."""
    out: set[tuple[int, int]] = set()
    for a, b in CONTACT_RE.findall(text):
        ia, ib = pos_to_seq.get(int(a)), pos_to_seq.get(int(b))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        out.add((min(ia, ib), max(ia, ib)))
    return out


def _in_band(pair: tuple[int, int], lo: int, hi: int | None) -> bool:
    s = abs(pair[0] - pair[1])
    return s >= lo and (hi is None or s <= hi)


def gt_by_band(gt: set[tuple[int, int]]) -> dict[str, set[tuple[int, int]]]:
    return {name: {g for g in gt if _in_band(g, lo, hi)} for name, (lo, hi) in BANDS.items()}


def score_rollout(pred: set[tuple[int, int]],
                  gtb: dict[str, set[tuple[int, int]]]) -> dict[str, float]:
    """Flat per-band metrics dict: ``{band}_npred/_tp/_prec/_rec/_f1``.

    recall is NaN for a band with no ground-truth contacts (undefined); f1 then
    NaN too. The ``all`` band always has GT here (targets are >= 5 contacts).
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
