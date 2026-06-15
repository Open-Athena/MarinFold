# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge the novelty axes into one per-candidate 2-D label + crosstabs.

Pulls together, per candidate:
- **fold novelty** (exp41 Foldseek verdict; ``*_vs_afdb_reps_similarity.csv``)
- **sequence leakage** vs training (``candidate_seq_leakage.csv``)
- **MSA depth** Neff tier (``candidate_msa_depth.csv``; optional, "pending"
  until ``fetch_msa_colabfold.py`` + ``msa_depth.py`` have run)
- **deposit date** (the temporal axis; from each source's ``*_manifest.csv``)

and writes ``data/candidate_2d_label.csv`` plus prints the headline crosstabs
(``notes/low-msa-eval-curation.md`` §6: the hardest cell is novel-fold ×
shallow-MSA). The join is by candidate: the Foldseek CSVs key on the structure
filename stem (``<pdb>`` for de novo/CAMEO, ``<domain>`` for CASP), which we map
back to each candidate via its ``pdb_id`` / ``stem``.
"""

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SIM_CSVS = {
    "denovo_pdb": DATA / "denovo_vs_afdb_reps_similarity.csv",
    "casp_fm": DATA / "casp_fm_vs_afdb_reps_similarity.csv",
    "cameo_hard": DATA / "cameo_hard_vs_afdb_reps_similarity.csv",
}
# Per-source manifests carry deposit_date (the temporal axis). Keyed by the
# candidate_sequences.csv source label -- not the manifest's own ``source``
# column (CASP records ``casp14_fm``/``casp15_fm`` there) -- so the three stems
# cross-listed in de novo *and* CAMEO each take their own source's date. CASP
# manifests have no deposition date (domains come from tarballs), so those rows
# stay blank.
MANIFEST_CSVS = {
    "denovo_pdb": DATA / "denovo_pdb_manifest.csv",
    "casp_fm": DATA / "casp_fm_manifest.csv",
    "cameo_hard": DATA / "cameo_hard_manifest.csv",
}


def load_fold_verdicts() -> dict[str, dict]:
    """sim-stem -> {verdict, best_train_qtmscore} across all source CSVs."""
    out: dict[str, dict] = {}
    for path in SIM_CSVS.values():
        if not path.exists():
            continue
        for r in csv.DictReader(path.open()):
            out[r["stem"]] = {
                "fold_verdict": r["verdict"],
                "best_train_qtm": r.get("best_train_qtmscore", ""),
            }
    return out


def load_keyed(path: Path, key: str = "stem") -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r[key]: r for r in csv.DictReader(path.open())}


def neff_tier(neff_str: str) -> str:
    """Bin Neff into curation tiers (notes/low-msa-eval-curation.md §2.3)."""
    if not neff_str:
        return "missing"
    n = float(neff_str)
    if n < 1.5:
        return "orphan"
    if n < 10:
        return "low"
    if n < 30:
        return "marginal"
    return "deep"


def fold_key(source: str, stem: str, pdb_id: str) -> str:
    """The Foldseek CSV stem for this candidate."""
    return stem if source == "casp_fm" else pdb_id


def crosstab(rows: list[dict], row_field: str, col_field: str,
             row_order: list[str], col_order: list[str]) -> str:
    import collections
    counts = collections.Counter((r[row_field], r[col_field]) for r in rows)
    seen_cols = [c for c in col_order if any((rk, c) in counts for rk in row_order)]
    width = max(len(r) for r in row_order) + 2
    head = " " * width + "".join(f"{c:>14}" for c in seen_cols) + f"{'total':>8}"
    lines = [head]
    for rk in row_order:
        cells = [counts.get((rk, c), 0) for c in seen_cols]
        if not sum(cells):
            continue
        lines.append(f"{rk:<{width}}" + "".join(f"{v:>14}" for v in cells) + f"{sum(cells):>8}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DATA / "candidate_2d_label.csv")
    args = p.parse_args()

    seqs = list(csv.DictReader((DATA / "candidate_sequences.csv").open()))
    folds = load_fold_verdicts()
    leak = load_keyed(DATA / "candidate_seq_leakage.csv")
    depth = load_keyed(DATA / "candidate_msa_depth.csv")
    have_depth = bool(depth)
    deposit = {src: load_keyed(path) for src, path in MANIFEST_CSVS.items()}

    rows = []
    for s in seqs:
        stem, source, pdb = s["stem"], s["source"], s["pdb_id"]
        f = folds.get(fold_key(source, stem, pdb), {})
        lk = leak.get(stem, {})
        dp = depth.get(stem, {})
        neff = dp.get("neff", "")
        tier = neff_tier(neff) if have_depth else "pending"
        rows.append({
            "dataset": source, "stem": stem, "pdb_id": pdb, "length": s["length"],
            "deposit_date": deposit.get(source, {}).get(stem, {}).get("deposit_date", ""),
            "fold_verdict": f.get("fold_verdict", "missing"),
            "nearest_train_fold_qtm": f.get("best_train_qtm", ""),
            "seq_leakage": lk.get("verdict", "missing"),
            "nearest_train_seq_fident": lk.get("best_train_fident", ""),
            "msa_n_seqs": dp.get("n_seqs", ""),
            "msa_neff": neff,
            "neff_tier": tier,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "stem", "pdb_id", "length", "deposit_date", "fold_verdict",
              "nearest_train_fold_qtm", "seq_leakage", "nearest_train_seq_fident",
              "msa_n_seqs", "msa_neff", "neff_tier"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}\n")

    fold_order = ["redundant", "same_fold", "novel_fold", "missing"]
    leak_order = ["redundant_seq", "novel_seq", "missing"]
    print("=== fold novelty (rows) x sequence leakage (cols) ===")
    print(crosstab(rows, "fold_verdict", "seq_leakage", fold_order, leak_order))
    if have_depth:
        tier_order = ["orphan", "low", "marginal", "deep", "missing"]
        print("\n=== fold novelty (rows) x MSA-depth tier (cols) ===")
        print(crosstab(rows, "fold_verdict", "neff_tier", fold_order, tier_order))
        print("\n=== sequence leakage (rows) x MSA-depth tier (cols) ===")
        print(crosstab(rows, "seq_leakage", "neff_tier", leak_order, tier_order))
    else:
        print("\n(MSA-depth tiers pending: run fetch_msa_colabfold.py + "
              "msa_depth.py, then re-run this.)")


if __name__ == "__main__":
    main()
