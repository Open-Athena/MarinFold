# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Which of #91's four gaps let the surviving contamination through?

The ESM-Atlas corpus *was* filtered — #91's funnel dropped 41,517 sequences at
≥40 % identity / ≥50 % coverage against exp65's `candidate_sequences.csv`. Tier
A nonetheless finds a large residue of contaminated rows in the published
corpus. That is only actionable if we can say *why* each surviving row survived,
because the four gaps have very different fixes.

So every dropped row is re-derived under **#91's own rule against #91's own
reference** — 40 % identity, 50 % query coverage, and only the 454 exp65
proteins, with FoldBench-100 excluded exactly as `create_dataset.sh` excluded
it — and assigned the first category that applies:

``foldbench_only``
    Nothing in exp65's 454 reaches it under any rule; only a FoldBench-100
    protein does. **Gap 1** — the benchmark's most contaminated slice was never
    in the reference.
``band_30_40``
    An exp65 protein covers ≥50 % of the query at 30–40 % identity, but nothing
    reaches 40 %. **Gap 2** — the band that was kept by design.
``remote_only``
    No protein reaches the identity-and-coverage bar at all; the row is caught
    only by the E ≤ 1e-3 significance arm. **Gap 2/3** — remote homology a
    coverage-gated identity filter cannot see.
``inside_91s_own_rule``
    An exp65 protein covers ≥50 % at ≥40 % identity — #91's stated rule should
    have removed this row. This is a different failure from a threshold being
    too loose and is worth separating out.

The categories are evaluated in that order and are mutually exclusive, so they
sum to the drop list.

**The rules really are comparable.** #91's funnel
(``pipeline.py:_drop_by_search``) ran ``mmseqs search --alignment-mode 3
--min-seq-id 0.40 -c 0.50 --cov-mode 1`` with the Atlas sequence as query and
the eval protein as target, so its coverage gate is over the *eval* protein —
the same side as our ``qcov``. The one parameter that differs is sensitivity:
``create_dataset.sh`` set ``SEARCH_SENSITIVITY="4.0"``, commented in its own
source as *"4.0 = faster, may miss a few near-40 %-id hits"*, where this run
uses exp65/#213's ``-s 7.5``. So ``inside_91s_own_rule`` should be read as the
measured price of that trade-off, not as a rule error.

    uv run python attribute_residual.py --work /data/exp225_decontam
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from decontam_lib import (
    ARMS,
    CORPORA,
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    is_sequence_contaminant,
    parse_target,
)
from sequence_droplist import FIELDS

HERE = Path(__file__).resolve().parent

#: #91's funnel threshold. Ours is SEQ_MIN_IDENTITY (0.30); the gap between the
#: two is the "band" category.
FUNNEL_IDENTITY = 0.40

#: The slice of the benchmark #91's `--eval-ref` never contained.
FOLDBENCH_DATASET = "foldbench100"

CATEGORY_FOLDBENCH = "foldbench_only"
CATEGORY_BAND = "band_30_40"
CATEGORY_REMOTE = "remote_only"
CATEGORY_INSIDE = "inside_91s_own_rule"
CATEGORIES = (CATEGORY_INSIDE, CATEGORY_BAND, CATEGORY_REMOTE, CATEGORY_FOLDBENCH)

CATEGORY_GAP = {
    CATEGORY_FOLDBENCH: "gap 1 — FoldBench-100 absent from the reference",
    CATEGORY_BAND: "gap 2 — the 30–40 % band kept by the 40 % threshold",
    CATEGORY_REMOTE: "gap 2/3 — remote homology below any identity bar",
    CATEGORY_INSIDE: "none — #91's own rule covers this row",
}


def classify(m8: Path, foldbench_keys: set[str], report_ceiling: float) -> dict:
    """One pass over the alignments; per dropped row, the best evidence of each kind."""
    # Per training row, three booleans are enough to decide the category.
    inside: set[tuple[str, str]] = set()      # exp65 hit at >=40% id, >=50% qcov
    band: set[tuple[str, str]] = set()        # exp65 hit at >=30% id, >=50% qcov
    exp65_any: set[tuple[str, str]] = set()   # any contaminating exp65 hit
    dropped: set[tuple[str, str]] = set()
    arms: dict[tuple[str, str], str] = {}

    with m8.open() as fh:
        for line in fh:
            hit = dict(zip(FIELDS, line.rstrip("\n").split("\t")))
            evalue = float(hit["evalue"])
            if evalue > report_ceiling:
                continue
            identity, qcov = float(hit["fident"]), float(hit["qcov"])
            if not is_sequence_contaminant(identity, qcov, evalue):
                continue

            target = parse_target(hit["target"])
            key = (target.arm, target.entry_id)
            dropped.add(key)
            arms[key] = target.arm
            if hit["query"] in foldbench_keys:
                continue
            exp65_any.add(key)
            if qcov >= SEQ_MIN_QCOV and identity >= FUNNEL_IDENTITY:
                inside.add(key)
            elif qcov >= SEQ_MIN_QCOV and identity >= SEQ_MIN_IDENTITY:
                band.add(key)

    counts: defaultdict[str, Counter] = defaultdict(Counter)
    for key in dropped:
        if key in inside:
            category = CATEGORY_INSIDE
        elif key in band:
            category = CATEGORY_BAND
        elif key in exp65_any:
            category = CATEGORY_REMOTE
        else:
            category = CATEGORY_FOLDBENCH
        counts[arms[key]][category] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--m8", type=Path, default=None, help="default: <work>/aln_all_hits.m8")
    ap.add_argument("--reference", type=Path, default=HERE / "data/reference/eval_structures.csv")
    ap.add_argument("--report-evalue-ceiling", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=HERE / "data/residual_attribution.csv")
    args = ap.parse_args()

    m8 = args.m8 or args.work / "aln_all_hits.m8"
    reference = pd.read_csv(args.reference)
    foldbench = set(reference.loc[reference["dataset"] == FOLDBENCH_DATASET, "key"])
    print(f"[attribute] {len(foldbench)} FoldBench-100 keys, "
          f"{len(reference) - len(foldbench)} exp65 keys (#91's reference)", flush=True)

    counts = classify(m8, foldbench, args.report_evalue_ceiling)

    rows = []
    for arm in ARMS:
        total = sum(counts[arm].values())
        for category in CATEGORIES:
            n = counts[arm][category]
            rows.append(
                {
                    "arm": arm,
                    "category": category,
                    "gap": CATEGORY_GAP[category],
                    "n_rows": n,
                    "pct_of_droplist": round(100 * n / total, 2) if total else 0.0,
                    "pct_of_corpus": round(100 * n / CORPORA[arm].n_documents, 4),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"  {row['arm']:<10} {row['category']:<22} {row['n_rows']:>9,} "
              f"({row['pct_of_droplist']:5.2f}% of the drop list)", flush=True)
    print(f"[attribute] -> {args.out}", flush=True)
    print(f"[note] identity thresholds: ours {SEQ_MIN_IDENTITY}, #91's {FUNNEL_IDENTITY}; "
          f"coverage {SEQ_MIN_QCOV}; significance {SEQ_MAX_EVALUE:g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
