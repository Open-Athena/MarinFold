# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6 — apply the label correction and recompute what it moves.

exp226's ``designed_any`` is a default, not a measurement, for every
``cameo_hard`` and ``casp_fm`` row: it resolved RCSB source organisms only for
the FoldBench monomers. :mod:`annotate_rcsb` resolved the rest and found 15 de
novo designs inside the 78 published as eval2-natural (and 4 more outside eval2).

This step turns that finding into the artifact downstream work should use:

* ``data/eval2_manifest_v2.csv`` — exp226's manifest, every original column
  preserved so it is a drop-in replacement, with ``designed_any`` corrected and
  the evidence carried alongside it (``designed_source``, ``kingdom``,
  ``entry_title``, ``escape_mechanism``). eval2-natural becomes **63**.
* ``data/eval2_headline_v2.csv`` / ``eval2_paired_deltas_v2.csv`` — exp226's
  scoreboard recomputed on the corrected split, plus viral / non-viral strata.
  **Moving the 15 changes the numbers, so they have to be recomputed**: the
  eval2-natural row and #226's "parity with Protenix single-seq comes back"
  finding were both computed with 15 designs inside the natural subset.
* ``data/correction_effect.csv`` — old vs new, side by side, so the size of the
  correction is explicit rather than implied.

The aggregation and the paired bootstrap are **imported from exp226**, not
reimplemented, so the v2 numbers differ from its published ones only by the
membership change (same seed, same 10,000 resamples, same estimator).

    uv run python apply_correction.py
"""
import argparse
import csv
import sys

import pandas as pd

import upstream as U

sys.path.insert(0, str(U.EXP226_DIR))
from build_eval2_scores import (  # noqa: E402
    CUTS,
    MARINFOLD,
    ORDER,
    RANGES,
    aggregate,
    paired_deltas,
)

DATA = U.HERE / "data"
EXP226_DATA = U.EXP226_DIR / "data"
PER_PROTEIN = EXP226_DATA / "eval2_per_protein.csv.gz"

OUT_MANIFEST = DATA / "eval2_manifest_v2.csv"
OUT_HEADLINE = DATA / "eval2_headline_v2.csv"
OUT_DELTAS = DATA / "eval2_paired_deltas_v2.csv"
OUT_EFFECT = DATA / "correction_effect.csv"

#: How each corrected row was decided, in the order the evidence is checked.
SOURCE_EXP226 = "exp226_dataset_or_foldbench_organism"
SOURCE_SYNTHETIC = "exp241_rcsb_synthetic_taxon"
SOURCE_DENOVO = "exp241_pdb_denovo_keyword"


def read(path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def build_manifest() -> pd.DataFrame:
    """exp226's manifest with the corrected flag and the evidence beside it."""
    manifest = read(U.EVAL2_MANIFEST)
    annotation = {(r["dataset"], r["stem"]): r
                  for r in read(DATA / "rcsb_annotation.csv")}
    mechanism = {(r["dataset"], r["stem"]): r
                 for r in read(DATA / "mechanism_table.csv")}

    rows = []
    for row in manifest:
        key = (row["dataset"], row["stem"])
        ann = annotation[key]
        published = int(row["designed_any"])
        synthetic = ann["is_synthetic_taxon"] == "1"
        denovo = ann["has_denovo_keyword"] == "1"
        corrected = int(published or synthetic or denovo)

        if published:
            source = SOURCE_EXP226
        elif synthetic:
            source = SOURCE_SYNTHETIC
        elif denovo:
            source = SOURCE_DENOVO
        else:
            source = ""

        out = dict(row)
        out["designed_any"] = corrected
        out["designed_any_exp226"] = published
        out["designed_source"] = source
        out["kingdom"] = ann["kingdom"]
        out["is_viral"] = int(ann["kingdom"] == "virus")
        out["uniprot_accessions"] = ann["uniprot_accessions"]
        out["in_afdb_full"] = mechanism.get(key, {}).get("in_afdb_full", "")
        out["escape_mechanism"] = mechanism.get(key, {}).get("escape_mechanism", "")
        out["entry_id"] = ann["entry_id"]
        out["entry_title"] = ann["title"]
        rows.append(out)

    frame = pd.DataFrame(rows)
    natural = int((frame["designed_any"] == 0).sum())
    if natural != U.EXPECTED_EVAL2_NATURAL_N - 15:
        raise SystemExit(
            f"corrected natural count is {natural}, expected "
            f"{U.EXPECTED_EVAL2_NATURAL_N - 15}")
    return frame


def score(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-aggregate exp226's six predictors under the corrected split."""
    wide = pd.read_csv(PER_PROTEIN)
    # The per-protein table carries exp226's flag; replace it with the audited
    # one, joining on the (dataset, stem) key exp226 itself uses. eval2 has 307
    # units over 305 unique stems, so the join must not be on stem alone.
    corrected = manifest[["dataset", "stem", "designed_any", "is_viral"]]
    wide = wide.drop(columns=["designed_any"]).merge(
        corrected, on=["dataset", "stem"], how="left", validate="many_to_one")
    if wide["designed_any"].isna().any():
        raise SystemExit("some scored proteins did not join to the manifest")

    natural = wide["designed_any"] == 0
    strict = wide["passes_30"] == 1
    viral = wide["is_viral"] == 1
    subsets = {
        "eval2 (<40% id)": pd.Series(True, index=wide.index),
        "eval2 natural (audited)": natural,
        "eval2 (<30% id)": strict,
        "eval2 natural (audited, <30%)": natural & strict,
        "eval2 natural, viral": natural & viral,
        "eval2 natural, non-viral": natural & ~viral,
    }
    return aggregate(wide, subsets), paired_deltas(wide, subsets)


def compare(headline: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """Old vs new for the rows the correction moves."""
    old_headline = pd.read_csv(EXP226_DATA / "eval2_headline.csv")
    old_deltas = pd.read_csv(EXP226_DATA / "eval2_paired_deltas.csv")
    rows = []

    pairs = [("eval2 natural", "eval2 natural (audited)"),
             ("eval2 natural (<30%)", "eval2 natural (audited, <30%)")]
    for old_label, new_label in pairs:
        for rng in RANGES:
            for cut in CUTS:
                a = old_headline[(old_headline["subset"] == old_label)
                                 & (old_headline["range"] == rng)
                                 & (old_headline["cut"] == cut)]
                b = headline[(headline["subset"] == new_label)
                             & (headline["range"] == rng)
                             & (headline["cut"] == cut)]
                if a.empty or b.empty:
                    continue
                for predictor in ORDER:
                    if predictor not in a or predictor not in b:
                        continue
                    rows.append({
                        "metric": f"{cut} ({rng})", "subset": new_label,
                        "predictor": predictor,
                        "n_published": int(a["n"].iloc[0]),
                        "n_audited": int(b["n"].iloc[0]),
                        "published": a[predictor].iloc[0],
                        "audited": b[predictor].iloc[0],
                        "change": round(float(b[predictor].iloc[0]
                                              - a[predictor].iloc[0]), 4),
                    })

    for old_label, new_label in pairs:
        a = old_deltas[old_deltas["subset"] == old_label]
        b = deltas[deltas["subset"] == new_label]
        for _, new_row in b.iterrows():
            old_row = a[a["baseline"] == new_row["baseline"]]
            if old_row.empty:
                continue
            old_row = old_row.iloc[0]
            rows.append({
                "metric": "paired delta (R, all)", "subset": new_label,
                "predictor": f"MarinFold - {new_row['baseline']}",
                "n_published": int(old_row["n"]), "n_audited": int(new_row["n"]),
                "published": f"{old_row['delta']:+.4f} "
                             f"[{old_row['ci_lo']:+.4f}, {old_row['ci_hi']:+.4f}]"
                             f"{'*' if old_row['significant'] else ''}",
                "audited": f"{new_row['delta']:+.4f} "
                           f"[{new_row['ci_lo']:+.4f}, {new_row['ci_hi']:+.4f}]"
                           f"{'*' if new_row['significant'] else ''}",
                "change": round(float(new_row["delta"] - old_row["delta"]), 4),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    manifest = build_manifest()
    manifest.to_csv(OUT_MANIFEST, index=False)
    n_designed = int((manifest["designed_any"] == 1).sum())
    print(f"[manifest] {OUT_MANIFEST.name}: {len(manifest)} rows, "
          f"{n_designed} designed ({n_designed / len(manifest):.0%}), "
          f"{len(manifest) - n_designed} natural "
          f"({int((manifest['is_viral'] == 1).sum())} viral overall)", flush=True)

    headline, deltas = score(manifest)
    headline.to_csv(OUT_HEADLINE, index=False)
    deltas.to_csv(OUT_DELTAS, index=False)

    effect = compare(headline, deltas)
    effect.to_csv(OUT_EFFECT, index=False)

    for _, row in headline[headline["cut"] == "R"].iterrows():
        cells = "  ".join(f"{p.split(' (')[0][:20]}={row[p]:.4f}"
                          for p in ORDER if p in row and pd.notna(row[p]))
        print(f"[R {row['range']:>4}] {row['subset']:<30} n={row['n']:<4} {cells}",
              flush=True)
    print(flush=True)
    for _, row in deltas[deltas["subset"].str.contains("natural")].iterrows():
        mark = "*" if row["significant"] else " "
        print(f"[delta {mark}] {row['subset']:<30} vs "
              f"{row['baseline']:<30} {row['delta']:+.4f} "
              f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] n={row['n']}", flush=True)
    print(f"\n[out] {OUT_HEADLINE.name}, {OUT_DELTAS.name}, {OUT_EFFECT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
