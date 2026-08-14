# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""What a pure ">= 30 % identity" rule costs, over one or several references.

Tier A is a disjunction — identity-and-coverage **or** E <= 1e-3 — and the
E-value arm is what catches remote homology. A narrower question is worth
pricing on its own: *drop every training protein at or above 30 % sequence
identity to anything in a reference*, with no significance arm at all. That is
the rule people reach for by default, and it is not the same list.

The second thing this prices is a **larger reference**. The pinned v1 reference
is the 554 proteins we report on, 100 of which are FoldBench monomers. FoldBench
itself is far bigger — protein-protein, antibody-antigen, protein-peptide,
protein-ligand, protein-DNA and protein-RNA tasks all carry protein chains, 1,940
of them (:mod:`foldbench_reference`). If any of those tasks might ever be
reported on, their chains have to be out of the corpus too.

So each reference is reduced independently and then unioned, and every
combination is reported: what each costs alone, and what the union costs. The
union is not the sum — the FoldBench monomers are in both references, and a
single training row is frequently homologous to several eval proteins at once.

Coverage
--------

Identity without a coverage gate is not homology — a 95 %-identical
twelve-residue match would otherwise delete a training protein — but *which*
sequence the coverage is measured over changes the answer, so it is an explicit
choice here rather than a convention.

In these alignments the **query is the eval/reference protein** and the
**target is the training protein**, so MMseqs2's ``qcov`` is the fraction of
the *eval* protein the alignment covers and ``tcov`` the fraction of the
*training* protein. The modes:

``shorter`` (default)
    ``max(qcov, tcov) >= 0.50`` — covers at least half of the **shorter** of
    the two sequences. For a fixed aligned region the shorter sequence always
    has the larger coverage, so the max *is* the shorter one's coverage. This
    catches a short training protein that matches one domain of a long eval
    protein, which a query-side gate misses entirely.
``reference``
    ``qcov >= 0.50`` — covers at least half of the *eval* protein. This is
    exp65's and #213's convention and what :mod:`sequence_droplist` applies for
    Tier A, so it is available here to reproduce those numbers exactly.
``training``
    ``tcov >= 0.50``.
``both``
    ``min(qcov, tcov) >= 0.50`` — the strictest, a near-global alignment.

    uv run python identity_droplist.py \\
        --reference eval554=/data/exp225_decontam/aln_all_hits.m8 \\
        --reference foldbench_all=/data/exp225_decontam/foldbench_all/aln_all_hits.m8
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from decontam_lib import (
    ARMS,
    CORPORA,
    REFERENCE_VERSION,
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    parse_target,
)
from sequence_droplist import FIELDS


#: How an alignment's ``(qcov, tcov)`` becomes the one number the gate applies
#: to. See the module docstring for what each means and why it matters.
COVERAGE_MODES = {
    "shorter": max,
    "reference": lambda qcov, tcov: qcov,
    "training": lambda qcov, tcov: tcov,
    "both": min,
}


def dropped_keys(
    m8: Path,
    *,
    min_identity: float,
    min_qcov: float,
    max_evalue: float | None,
    report_ceiling: float = float("inf"),
    coverage_mode: str = "shorter",
) -> dict[str, set[str]]:
    """``{arm: {entry_id, ...}}`` for rows this reference's rule removes.

    ``max_evalue`` is the optional significance arm: ``None`` applies the pure
    identity rule, a float unions in Tier A's ``E <= 1e-3`` catch-all.

    ``report_ceiling`` discards alignments mmseqs reported below the depth the
    tier is defined at. It matters only for the identity arm, which has no
    significance floor of its own — the searches here run at ``-e 1000`` so the
    sensitivity can be swept, and reducing all of that would silently price a
    different rule than :mod:`sequence_droplist` does.
    """
    combine = COVERAGE_MODES[coverage_mode]
    dropped: dict[str, set[str]] = {arm: set() for arm in ARMS}
    with m8.open() as fh:
        for line in fh:
            hit = dict(zip(FIELDS, line.rstrip("\n").split("\t")))
            evalue = float(hit["evalue"])
            if evalue > report_ceiling:
                continue
            identity = float(hit["fident"])
            coverage = combine(float(hit["qcov"]), float(hit["tcov"]))
            by_identity = identity >= min_identity and coverage >= min_qcov
            by_evalue = max_evalue is not None and evalue <= max_evalue
            if by_identity or by_evalue:
                target = parse_target(hit["target"])
                dropped[target.arm].add(target.entry_id)
    return dropped


def summarise(label: str, dropped: dict[str, set[str]], rule: str) -> list[dict]:
    rows = []
    for arm in ARMS:
        corpus = CORPORA[arm]
        n = len(dropped[arm])
        rows.append(
            {
                "reference": label,
                "rule": rule,
                "arm": arm,
                "n_documents": corpus.n_documents,
                "n_dropped": n,
                "pct_dropped": round(100 * n / corpus.n_documents, 4),
                "n_surviving": corpus.n_documents - n,
            }
        )
    return rows


def parse_reference(text: str) -> tuple[str, Path]:
    label, _, path = text.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reference", type=parse_reference, action="append", required=True,
                    metavar="LABEL=ALIGNMENTS.m8",
                    help="a reference's alignment file, repeatable")
    ap.add_argument("--min-identity", type=float, default=SEQ_MIN_IDENTITY)
    ap.add_argument("--min-coverage", type=float, default=SEQ_MIN_QCOV,
                    dest="min_qcov")
    ap.add_argument("--coverage-mode", choices=sorted(COVERAGE_MODES), default="shorter",
                    help="which sequence the coverage gate is measured over "
                         "(see the module docstring)")
    ap.add_argument("--with-evalue-arm", action="store_true",
                    help=f"also drop on E <= {SEQ_MAX_EVALUE:g}, i.e. the full Tier A rule")
    ap.add_argument("--report-evalue-ceiling", type=float, default=float("inf"),
                    help="ignore alignments above this E-value. Default is no ceiling: "
                         "the rule is identity+coverage only. Set 10 (exp65's and #213's) "
                         "to reproduce the Tier A tables exactly")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "data/identity_droplist.csv")
    args = ap.parse_args()

    max_evalue = SEQ_MAX_EVALUE if args.with_evalue_arm else None
    rule = (
        f"identity >= {args.min_identity:.0%} over >= {args.min_qcov:.0%} of the "
        f"{args.coverage_mode} sequence"
        + (f" or E <= {SEQ_MAX_EVALUE:g}" if max_evalue else "")
        + (f" (alignments reduced only to E <= {args.report_evalue_ceiling:g})"
           if args.report_evalue_ceiling != float("inf") else "")
    )
    print(f"[rule] {rule}", flush=True)

    per_reference: dict[str, dict[str, set[str]]] = {}
    rows: list[dict] = []
    for label, m8 in args.reference:
        if not m8.exists():
            raise SystemExit(f"{label}: {m8} does not exist")
        dropped = dropped_keys(
            m8, min_identity=args.min_identity, min_qcov=args.min_qcov,
            max_evalue=max_evalue, report_ceiling=args.report_evalue_ceiling,
            coverage_mode=args.coverage_mode,
        )
        per_reference[label] = dropped
        rows += summarise(label, dropped, rule)
        for arm in ARMS:
            print(f"  {label:<16} {arm:<10} {len(dropped[arm]):>9,} "
                  f"({100 * len(dropped[arm]) / CORPORA[arm].n_documents:6.3f}%)", flush=True)

    labels = list(per_reference)
    for size in range(2, len(labels) + 1):
        for combo in combinations(labels, size):
            union = {
                arm: set().union(*(per_reference[label][arm] for label in combo))
                for arm in ARMS
            }
            label = " + ".join(combo)
            rows += summarise(label, union, rule)
            for arm in ARMS:
                print(f"  {label:<16} {arm:<10} {len(union[arm]):>9,} "
                      f"({100 * len(union[arm]) / CORPORA[arm].n_documents:6.3f}%)",
                      flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[identity] -> {args.out}", flush=True)

    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "reference_version": REFERENCE_VERSION,
                "rule": rule,
                "min_identity": args.min_identity,
                "min_coverage": args.min_qcov,
                "coverage_mode": args.coverage_mode,
                "search_reported_to_evalue": 1000.0,
                "evalue_arm": SEQ_MAX_EVALUE if max_evalue else None,
                "report_evalue_ceiling": args.report_evalue_ceiling,
                "alignments": {label: str(path) for label, path in args.reference},
                "totals": {f"{r['reference']}/{r['arm']}": r["n_dropped"] for r in rows},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
