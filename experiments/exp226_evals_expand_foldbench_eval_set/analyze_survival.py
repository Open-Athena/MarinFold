# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 — survival counts at <40 % / <30 % identity, and the newer-vs-older test.

Consumes :mod:`search_expanded`'s 776-row identity table and answers the four
things issue #226 asks for:

1. **Survival**, at both identity filters, split by dataset and by designed vs
   natural. Reported on the coverage-gated identity axis (exp213's rule) with
   the ungated maximum alongside as the paranoid bound.
2. **Designed vs natural, properly.** exp213 splits on the dataset label
   (``denovo_pdb``), which cannot see a designed protein sitting in a FoldBench
   row — and 12 FoldBench monomers are literally in exp65's de novo set, so some
   do. :mod:`build_query_set` resolves each FoldBench entity's source organism;
   this joins that flag on, so "natural" means *no known design*, not merely
   "not in the de novo dataset".
3. **Do the newer 234 behave like the older 100?** — the question the issue's
   extrapolation could not answer, since our 100 are the oldest-deposited rows
   rather than a random sample. Tested with a two-sided Fisher exact test on the
   survival counts, plus the length profiles the extrapolation leaned on.
4. **Per-arm attribution** of the proteins that get filtered out.

    uv run python analyze_survival.py
"""
import argparse
import csv
import json
import statistics
from math import comb
from pathlib import Path

from exp213_link import ARMS, EXP213_TABLE

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: The two sequence-identity filters the issue asks about.
THRESHOLDS = (0.40, 0.30)

#: Dataset labels, in report order. ``foldbench_rest`` is this experiment's
#: addition; the other four are the 554-protein eval set.
DATASETS = ("foldbench100", "foldbench_rest", "denovo_pdb", "cameo_hard", "casp_fm")
DATASET_NEW = "foldbench_rest"

#: Identity columns: the coverage-gated axis, and the ungated paranoid bound.
GATED = "best_identity_covered"
UNGATED = "best_identity_any"


def survives(row: dict, threshold: float, column: str = GATED) -> bool:
    """True when ``row``'s nearest training sequence is below ``threshold``.

    A protein with no covered hit at all (empty column) survives every filter —
    it has no measurable training relative on that axis.
    """
    value = row[column]
    return value == "" or float(value) < threshold


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table ``[[a, b], [c, d]]``.

    Exact rather than a chi-square approximation because the survivor counts
    here are small (11-23), which is precisely where the approximation fails.
    Implemented directly — the experiment's dependency set is numpy/pandas/
    matplotlib and one hypergeometric sum does not justify adding scipy.
    """
    row1, row2 = a + b, c + d
    col1, total = a + c, a + b + c + d

    def p(k: int) -> float:
        return comb(row1, k) * comb(row2, col1 - k) / comb(total, col1)

    observed = p(a)
    low = max(0, col1 - row2)
    high = min(row1, col1)
    # Sum every table at least as extreme as the observed one, with a relative
    # tolerance so float noise cannot drop the observed table from its own sum.
    return min(1.0, sum(p(k) for k in range(low, high + 1)
                        if p(k) <= observed * (1 + 1e-9)))


def load_rows(table: Path, targets: Path) -> list[dict]:
    """The identity table, with the FoldBench source-organism flag joined on.

    ``designed`` is exp213's dataset rule, carried through unchanged so the two
    tables agree. ``synthetic`` is the source-organism flag, available only for
    FoldBench rows (the only ones :mod:`build_query_set` resolves against RCSB);
    it is empty elsewhere. ``designed_any`` is the union — the honest split.
    """
    by_stem = {r["stem"]: r for r in csv.DictReader(targets.open())}
    rows = list(csv.DictReader(table.open()))
    for row in rows:
        foldbench = row["dataset"] in ("foldbench100", DATASET_NEW)
        synthetic = by_stem[row["stem"]]["synthetic"] == "1" if foldbench else None
        row["synthetic"] = "" if synthetic is None else int(synthetic)
        row["designed_any"] = int(row["designed"] == "1" or bool(synthetic))
        row["source_names"] = by_stem[row["stem"]]["source_names"] if foldbench else ""
    return rows


def survival_by_dataset(rows: list[dict]) -> list[dict]:
    out = []
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        entry = {"dataset": dataset, "n": len(subset),
                 "n_designed_any": sum(r["designed_any"] for r in subset)}
        for threshold in THRESHOLDS:
            tag = f"{threshold:.0%}".rstrip("%")
            survivors = [r for r in subset if survives(r, threshold)]
            entry[f"survive_{tag}"] = len(survivors)
            entry[f"survive_{tag}_pct"] = round(100 * len(survivors) / len(subset), 1)
            entry[f"survive_{tag}_natural"] = sum(1 for r in survivors
                                                  if not r["designed_any"])
            entry[f"survive_{tag}_ungated"] = sum(1 for r in subset
                                                  if survives(r, threshold, UNGATED))
        out.append(entry)
    return out


def headline(rows: list[dict]) -> list[dict]:
    """The before/after table: the 554 eval set vs the 776 expanded one."""
    original = [r for r in rows if r["dataset"] != DATASET_NEW]
    out = []
    for threshold in THRESHOLDS:
        row = {"threshold": f"<{threshold:.0%}"}
        for label, subset in (("orig554", original), ("expanded776", rows)):
            survivors = [r for r in subset if survives(r, threshold)]
            natural = [r for r in survivors if not r["designed_any"]]
            row[f"{label}_n"] = len(subset)
            row[f"{label}_survive"] = len(survivors)
            row[f"{label}_natural"] = len(natural)
            row[f"{label}_designed"] = len(survivors) - len(natural)
            row[f"{label}_survive_ungated"] = sum(1 for r in subset
                                                  if survives(r, threshold, UNGATED))
        row["gain_survive"] = row["expanded776_survive"] - row["orig554_survive"]
        row["gain_natural"] = row["expanded776_natural"] - row["orig554_natural"]
        row["gain_natural_pct"] = round(
            100 * row["gain_natural"] / row["orig554_natural"], 1)
        out.append(row)
    return out


def newer_vs_older(rows: list[dict]) -> list[dict]:
    """Does the unused 222 survive at the same rate as the 100 we already use?

    Our 100 are the *first* 100 rows of a roughly PDB-id-sorted file — the
    oldest-deposited entries, not a random draw — so the issue's extrapolation
    from their survival rate is exactly what needs checking.
    """
    old = [r for r in rows if r["dataset"] == "foldbench100"]
    new = [r for r in rows if r["dataset"] == DATASET_NEW]
    out = []
    for threshold in THRESHOLDS:
        old_s = sum(1 for r in old if survives(r, threshold))
        new_s = sum(1 for r in new if survives(r, threshold))
        predicted = round(len(new) * old_s / len(old), 1)
        out.append({
            "threshold": f"<{threshold:.0%}",
            "old_n": len(old), "old_survive": old_s,
            "old_pct": round(100 * old_s / len(old), 1),
            "new_n": len(new), "new_survive": new_s,
            "new_pct": round(100 * new_s / len(new), 1),
            "predicted_from_old_rate": predicted,
            "shortfall": round(new_s - predicted, 1),
            "fisher_p": round(fisher_exact_two_sided(
                old_s, len(old) - old_s, new_s, len(new) - new_s), 4),
        })
    return out


def length_profiles(rows: list[dict]) -> dict:
    old = [int(r["query_len"]) for r in rows if r["dataset"] == "foldbench100"]
    new = [int(r["query_len"]) for r in rows if r["dataset"] == DATASET_NEW]
    return {
        "foldbench100": {"n": len(old), "median": statistics.median(old),
                         "mean": round(statistics.mean(old), 1),
                         "min": min(old), "max": max(old)},
        "foldbench_rest": {"n": len(new), "median": statistics.median(new),
                           "mean": round(statistics.mean(new), 1),
                           "min": min(new), "max": max(new)},
    }


def survival_by_arm(rows: list[dict]) -> list[dict]:
    """Survival under each training arm alone, and under the union.

    exp199 was trained on **both** corpora, so the union is the filter that
    matters and it is what the headline reports. But the two arms are worth
    separating, because every prior overlap analysis (#41, #65, #94) only ever
    checked AFDB — and the AFDB-only column is what those would have concluded.
    """
    subsets = (("net_new222", [r for r in rows if r["dataset"] == DATASET_NEW]),
               ("foldbench100", [r for r in rows if r["dataset"] == "foldbench100"]),
               ("orig554", [r for r in rows if r["dataset"] != DATASET_NEW]),
               ("expanded776", rows))
    filters = (("afdb_only", f"{ARMS[0]}_{GATED}"),
               ("esm_atlas_only", f"{ARMS[1]}_{GATED}"),
               ("union", GATED))
    out = []
    for name, subset in subsets:
        for threshold in THRESHOLDS:
            entry = {"subset": name, "n": len(subset),
                     "threshold": f"<{threshold:.0%}"}
            for label, column in filters:
                survivors = [r for r in subset if survives(r, threshold, column)]
                entry[f"survive_{label}"] = len(survivors)
                entry[f"natural_{label}"] = sum(1 for r in survivors
                                                if not r["designed_any"])
            entry["esm_atlas_removes_beyond_afdb"] = (
                entry["survive_afdb_only"] - entry["survive_union"])
            entry["afdb_removes_beyond_esm_atlas"] = (
                entry["survive_esm_atlas_only"] - entry["survive_union"])
            out.append(entry)
    return out


def arm_complementarity(rows: list[dict]) -> list[dict]:
    """Which arm supplies the >=threshold homolog that filters a protein out.

    Distinct from :func:`arm_attribution`, which asks only whether an arm has
    *any* significant hit (exp213's statistic). This asks whether an arm has a
    hit at or above the identity threshold — i.e. whether that arm alone would
    have been enough to remove the protein.
    """
    def blocks(row: dict, arm: str, threshold: float) -> bool:
        return not survives(row, threshold, f"{arm}_{GATED}")

    subsets = (("net_new222", [r for r in rows if r["dataset"] == DATASET_NEW]),
               ("orig554", [r for r in rows if r["dataset"] != DATASET_NEW]),
               ("expanded776", rows))
    out = []
    for name, subset in subsets:
        for threshold in THRESHOLDS:
            afdb = [r for r in subset if blocks(r, ARMS[0], threshold)]
            esm = [r for r in subset if blocks(r, ARMS[1], threshold)]
            both = [r for r in subset if blocks(r, ARMS[0], threshold)
                    and blocks(r, ARMS[1], threshold)]
            out.append({
                "subset": name, "n": len(subset), "threshold": f"<{threshold:.0%}",
                "dropped": len(afdb) + len(esm) - len(both),
                "both_arms": len(both),
                "afdb_only": len(afdb) - len(both),
                "esm_atlas_only": len(esm) - len(both),
            })
    return out


def arm_attribution(rows: list[dict], threshold: float) -> list[dict]:
    """Which training arm supplies the homology that filters a protein out."""
    out = []
    for label, subset in (("orig554", [r for r in rows if r["dataset"] != DATASET_NEW]),
                          ("net_new222", [r for r in rows if r["dataset"] == DATASET_NEW]),
                          ("expanded776", rows)):
        dropped = [r for r in subset if not survives(r, threshold)]
        hits = {arm: [int(r[f"{arm}_n_hits_significant"]) > 0 for r in dropped]
                for arm in ARMS}
        both = sum(1 for a, e in zip(*hits.values()) if a and e)
        out.append({
            "subset": label, "threshold": f"<{threshold:.0%}",
            "n": len(subset), "dropped": len(dropped),
            "both_arms": both,
            "afdb_only": sum(hits[ARMS[0]]) - both,
            "esm_atlas_only": sum(hits[ARMS[1]]) - both,
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {len(rows)} rows -> {path}", flush=True)


def check_parity(rows: list[dict], exp213_table: Path) -> dict:
    """The 554 must reproduce exp213's table exactly, or nothing else compares."""
    old = {(r["dataset"], r["stem"]): r for r in csv.DictReader(exp213_table.open())}
    shared = [r for r in rows if (r["dataset"], r["stem"]) in old]
    if len(shared) != len(old):
        raise SystemExit(f"expanded table covers {len(shared)}/{len(old)} of exp213's rows")
    changed = [r["stem"] for r in shared
               if r[GATED] != old[(r["dataset"], r["stem"])][GATED]
               or r["stratum"] != old[(r["dataset"], r["stem"])]["stratum"]]
    if changed:
        raise SystemExit(
            f"{len(changed)} of exp213's rows changed identity or stratum in the "
            f"expanded search ({changed[:5]}); the parameters are not exp213's."
        )
    parity = {}
    for column, expected in ((GATED, {0.40: 284, 0.30: 264}),
                             (UNGATED, {0.40: 273, 0.30: 255})):
        for threshold, want in expected.items():
            got = sum(1 for r in shared if survives(r, threshold, column))
            key = f"{column}_{threshold:.0%}".rstrip("%")
            parity[key] = {"got": got, "expected": want}
            if got != want:
                raise SystemExit(
                    f"554-subset survival at <{threshold:.0%} on {column} is {got}, "
                    f"expected exp213's {want} — parameter parity is broken."
                )
    print(f"[parity] 554 subset reproduces exp213 exactly: 284/264 gated, "
          f"273/255 ungated, 0/{len(shared)} rows changed", flush=True)
    return {"rows_compared": len(shared), "rows_changed": 0, "counts": parity}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=DATA / "eval_train_identity_expanded.csv")
    ap.add_argument("--targets", type=Path, default=DATA / "foldbench_targets.csv")
    ap.add_argument("--exp213-table", type=Path, default=EXP213_TABLE)
    args = ap.parse_args()

    rows = load_rows(args.table, args.targets)
    parity = check_parity(rows, args.exp213_table)

    by_dataset = survival_by_dataset(rows)
    head = headline(rows)
    newer = newer_vs_older(rows)
    arms = [e for t in THRESHOLDS for e in arm_attribution(rows, t)]
    by_arm = survival_by_arm(rows)
    complementarity = arm_complementarity(rows)

    write_csv(by_dataset, DATA / "survival_by_dataset.csv")
    write_csv(head, DATA / "survival_headline.csv")
    write_csv(newer, DATA / "newer_vs_older.csv")
    write_csv(arms, DATA / "arm_attribution.csv")
    write_csv(by_arm, DATA / "survival_by_arm.csv")
    write_csv(complementarity, DATA / "arm_complementarity.csv")

    summary = {
        "parity_with_exp213": parity,
        "expanded_set_size": len(rows),
        "headline": head,
        "newer_vs_older": newer,
        "length_profiles": length_profiles(rows),
        "survival_by_arm": by_arm,
        "arm_complementarity": complementarity,
        "designed_flag": {
            "note": "exp213's `designed` is the denovo_pdb dataset label; "
                    "`synthetic` is RCSB source-organism (taxid 32630 / no natural "
                    "source), resolved only for FoldBench rows; `designed_any` is "
                    "their union.",
            "foldbench100_synthetic": sum(1 for r in rows
                                          if r["dataset"] == "foldbench100" and r["synthetic"] == 1),
            "foldbench_rest_synthetic": sum(1 for r in rows
                                            if r["dataset"] == DATASET_NEW and r["synthetic"] == 1),
        },
    }
    (DATA / "survival_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[write] -> {DATA / 'survival_summary.json'}", flush=True)

    for row in head:
        print(f"[headline] {row['threshold']}: "
              f"{row['orig554_survive']}/554 -> {row['expanded776_survive']}/776 "
              f"({row['gain_survive']:+d}); natural {row['orig554_natural']} -> "
              f"{row['expanded776_natural']} ({row['gain_natural']:+d}, "
              f"{row['gain_natural_pct']:+.0f}%)", flush=True)
    for row in newer:
        print(f"[newer-vs-older] {row['threshold']}: old {row['old_pct']}% vs new "
              f"{row['new_pct']}%; extrapolation predicted "
              f"{row['predicted_from_old_rate']}, got {row['new_survive']} "
              f"({row['shortfall']:+.1f}); Fisher p={row['fisher_p']}", flush=True)
    for row in by_arm:
        if row["subset"] != "net_new222":
            continue
        print(f"[arms] net-new 222 {row['threshold']}: AFDB-only would leave "
              f"{row['survive_afdb_only']}, ESM-Atlas-only {row['survive_esm_atlas_only']}, "
              f"union {row['survive_union']} "
              f"(ESM-Atlas removes {row['esm_atlas_removes_beyond_afdb']} that AFDB "
              f"alone would have kept)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
