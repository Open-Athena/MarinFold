# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""How much does Tier A's size depend on how deep MMseqs2 was asked to report?

Tier A is a disjunction — ``E <= 1e-3`` **or** ``identity >= 30 % over >= 50 %
query coverage`` — and the two arms behave completely differently under the
reporting threshold. The E-value arm is self-limiting: nothing above 1e-3
qualifies, so reporting deeper adds nothing. The identity arm has no such
floor, and for a short query, 30 % identity over half of it is reachable by
chance, so the deeper mmseqs reports the more rows that arm removes. Left
unmeasured, "Tier A costs 1.8 % of AFDB" would be a number about an mmseqs
flag as much as about contamination.

This reduces the *same* alignment file at a ladder of ceilings and reports the
resulting corpus cost. Reading a flat curve means the threshold is not
load-bearing and the tier is well posed; reading a curve that keeps climbing
means the identity arm is absorbing noise and the honest headline has to name
the ceiling it was computed at.

    uv run python sweep_evalue.py --work /data/exp225_decontam
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from decontam_lib import ARMS, CORPORA, SEQ_MAX_EVALUE
from sequence_droplist import RULE_IDENTITY, build_droplist

HERE = Path(__file__).resolve().parent

#: 1e-3 is the E-value arm alone (the identity arm contributes nothing below
#: it); 10 is exp65's and #213's, and the tier's canonical ceiling; the rest
#: bound how far the identity arm can run.
CEILINGS = (SEQ_MAX_EVALUE, 1.0, 10.0, 100.0, 1000.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--m8", type=Path, default=None, help="default: <work>/aln_all_hits.m8")
    ap.add_argument("--ceilings", type=float, nargs="+", default=list(CEILINGS))
    ap.add_argument("--out", type=Path, default=HERE / "data/evalue_sensitivity.csv")
    args = ap.parse_args()

    m8 = args.m8 or args.work / "aln_all_hits.m8"
    if not m8.exists():
        raise SystemExit(f"{m8} does not exist; run sequence_droplist.py first")

    rows = []
    for ceiling in sorted(args.ceilings):
        dropped, stats = build_droplist(m8, ceiling)
        by_arm = {arm: 0 for arm in ARMS}
        identity_only = 0
        for record in dropped.values():
            by_arm[record["arm"]] += 1
            identity_only += record["rule"] == RULE_IDENTITY
        row = {
            "report_evalue_ceiling": ceiling,
            "n_alignments_considered": stats["n_alignments"],
            "n_dropped_total": len(dropped),
            "n_dropped_identity_arm_only": identity_only,
        }
        for arm in ARMS:
            corpus = CORPORA[arm]
            row[f"{arm}_n_dropped"] = by_arm[arm]
            row[f"{arm}_pct_dropped"] = round(100 * by_arm[arm] / corpus.n_documents, 4)
        rows.append(row)
        print(
            f"[sweep] E <= {ceiling:<8g} "
            + "  ".join(f"{arm}: {row[f'{arm}_pct_dropped']:.3f}%" for arm in ARMS)
            + f"  (identity-arm-only rows: {identity_only:,})",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
