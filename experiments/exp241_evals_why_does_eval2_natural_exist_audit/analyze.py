# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 — assign each of the 78 a mechanism, and cross-tab the rest. (A1/A2/A4/A6)

Consumes :mod:`annotate_rcsb`'s and :mod:`check_training_reachability`'s tables
and emits the experiment's result CSVs. Nothing here calls the network.

The mechanism ladder is deliberately ordered so that each protein is charged to
the *earliest* explanation that applies, and every rung is a check that was run
rather than an inference:

``designed_not_natural``  a synthetic-construct source taxon or a DE NOVO
                          PROTEIN keyword — the protein is not natural, so it
                          never belonged in the natural subset (A2).
``not_in_uniprot``        natural, but the deposited entity carries no UniProt
                          cross-reference: an engineered receptor, a chimera, a
                          hypervariable immune chain, or a metagenomic sequence
                          that was never deposited as a UniProt entry. Its
                          sequence is not in the reference databases either, so
                          no corpus built from them could contain it.
``afdb_absent``           in UniProt, but AFDB has no model for the accession —
                          AlphaFold never folded it, so the AFDB arm could not
                          have held it at any sampling rate.
``unsampled_corpus``      AFDB *does* have a model and we simply did not train
                          on it. The corpus is a 1.9 % sample of AFDB; this is
                          the arithmetic of that sample.
``search_miss``           the accession is in the training arm and the search
                          still reported < 40 % identity. This would be a
                          pipeline defect; the count is expected to be 0 and is
                          reported so that it is checked rather than assumed.
"""
import csv
import json
from collections import Counter

import upstream as U

DATA = U.HERE / "data"
ANNOTATION = DATA / "rcsb_annotation.csv"
REACHABILITY = DATA / "training_reachability.csv"

#: Eval sources whose proteins were selected *because* they are hard or
#: template-free, versus sources that are a slice of recent PDB.
NOVELTY_CURATED = {"casp_fm", "cameo_hard"}
RECENT_PDB = {"foldbench100", "foldbench_rest"}

MECHANISMS = ["designed_not_natural", "not_in_uniprot", "afdb_absent",
              "unsampled_corpus", "search_miss"]


def read_csv(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def by_key(rows):
    return {(r["dataset"], r["stem"]): r for r in rows}


def as_int(value, default=None):
    return int(value) if value not in ("", None) else default


def as_float(value, default=None):
    return float(value) if value not in ("", None) else default


def classify(reach: dict) -> str:
    if as_int(reach["designed_signal"], 0):
        return "designed_not_natural"
    if as_int(reach["in_afdb_arm"], 0):
        return "search_miss"
    if not reach["uniprot_accessions"]:
        return "not_in_uniprot"
    if not as_int(reach["in_afdb_full"], 0):
        return "afdb_absent"
    return "unsampled_corpus"


def write(path, rows, fieldnames=None):
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] {path.name} ({len(rows)} rows)")


def main() -> int:
    proteins = U.read_proteins()
    natural = U.eval2_natural(proteins)
    annotation = by_key(read_csv(ANNOTATION))
    reachability = by_key(read_csv(REACHABILITY))
    dates = U.read_exp65_dates()
    summary: dict[str, object] = {}

    # --- A1: where the 78 come from ---------------------------------------
    provenance = []
    for dataset in sorted({p.dataset for p in natural}):
        subset = [p for p in natural if p.dataset == dataset]
        provenance.append({
            "dataset": dataset,
            "n": len(subset),
            "selection": ("novelty-curated" if dataset in NOVELTY_CURATED
                          else "recent PDB"),
            "n_no_significant_homolog": sum(
                1 for p in subset if p.n_hits_significant == 0),
            "median_length": sorted(p.length for p in subset)[len(subset) // 2],
        })
    provenance.append({
        "dataset": "TOTAL", "n": len(natural), "selection": "",
        "n_no_significant_homolog": sum(
            1 for p in natural if p.n_hits_significant == 0),
        "median_length": sorted(p.length for p in natural)[len(natural) // 2],
    })
    write(DATA / "provenance_of_the_78.csv", provenance)

    # --- A2: the label audit, over the whole eval universe ------------------
    audit_rows, corrected = [], []
    for p in proteins:
        ann = annotation[(p.dataset, p.stem)]
        signal = (as_int(ann["is_synthetic_taxon"], 0)
                  or as_int(ann["has_denovo_keyword"], 0))
        designed_now = bool(p.designed_any or signal)
        if p.in_eval2 and not designed_now:
            corrected.append(p)
        if p.designed_any != designed_now:
            audit_rows.append({
                "dataset": p.dataset, "stem": p.stem,
                "entry_id": ann["entry_id"],
                "published_designed_any": int(p.designed_any),
                "audited_designed": int(designed_now),
                "is_synthetic_taxon": ann["is_synthetic_taxon"],
                "has_denovo_keyword": ann["has_denovo_keyword"],
                "has_uniprot_xref": ann["has_uniprot_xref"],
                "source_organisms": ann["source_organisms"],
                "in_eval2": int(p.in_eval2),
                "title": ann["title"],
            })
    write(DATA / "label_audit.csv", audit_rows)
    summary["eval2_natural_published"] = len(natural)
    summary["eval2_natural_audited"] = len(corrected)
    summary["eval2_reclassified_designed"] = sum(
        1 for r in audit_rows if as_int(r["in_eval2"], 0))

    # --- A3/A2 combined: the mechanism ladder -------------------------------
    mechanism_rows = []
    for p in natural:
        ann = annotation[(p.dataset, p.stem)]
        reach = reachability[(p.dataset, p.stem)]
        mechanism_rows.append({
            "dataset": p.dataset, "stem": p.stem,
            "selection": ("novelty-curated" if p.dataset in NOVELTY_CURATED
                          else "recent PDB"),
            "length": p.length, "kingdom": ann["kingdom"],
            "source_organisms": ann["source_organisms"],
            "escape_mechanism": classify(reach),
            "best_identity": reach["best_identity"],
            "n_hits_significant": p.n_hits_significant,
            "msa_neff": reach["msa_neff"],
            "uniprot_accessions": reach["uniprot_accessions"],
            "in_afdb_full": reach["in_afdb_full"],
            "in_afdb_arm": reach["in_afdb_arm"],
            "uniref50_size": reach["uniref50_size"],
            "uniref50_in_arm": reach["uniref50_in_arm"],
            "uniref90_size": reach["uniref90_size"],
            "uniref90_in_arm": reach["uniref90_in_arm"],
            "deposit_date": ann["deposit_date"] or dates.get(p.stem, ""),
            "uniprot_first_public": reach["uniprot_first_public"],
            "title": ann["title"],
        })
    write(DATA / "mechanism_table.csv", mechanism_rows)

    counts = Counter(r["escape_mechanism"] for r in mechanism_rows)
    mech_counts = [{
        "escape_mechanism": m, "n": counts.get(m, 0),
        "share": f"{counts.get(m, 0) / len(mechanism_rows):.4f}",
        "n_novelty_curated": sum(
            1 for r in mechanism_rows
            if r["escape_mechanism"] == m and r["selection"] == "novelty-curated"),
        "n_recent_pdb": sum(
            1 for r in mechanism_rows
            if r["escape_mechanism"] == m and r["selection"] == "recent PDB"),
    } for m in MECHANISMS]
    write(DATA / "mechanism_counts.csv", mech_counts)

    # The headline of A3, restricted to the proteins that are actually natural.
    real = [r for r in mechanism_rows
            if r["escape_mechanism"] != "designed_not_natural"]
    summary["natural_after_audit"] = len(real)
    summary["natural_with_uniprot_xref"] = sum(
        1 for r in real if r["uniprot_accessions"])
    summary["natural_with_afdb_model"] = sum(
        1 for r in real if as_int(r["in_afdb_full"], 0))
    summary["natural_in_afdb_arm"] = sum(
        1 for r in real if as_int(r["in_afdb_arm"], 0))
    sizes = [as_int(r["uniref50_size"]) for r in real if r["uniref50_size"]]
    summary["uniref50_reported"] = len(sizes)
    summary["uniref50_singletons"] = sum(1 for s in sizes if s == 1)
    summary["uniref50_median_size"] = sorted(sizes)[len(sizes) // 2] if sizes else None
    summary["uniref50_total_relatives"] = sum(sizes)
    summary["uniref50_relatives_in_arm"] = sum(
        as_int(r["uniref50_in_arm"], 0) for r in real)

    # --- A4: per-arm identity profile and the kingdom gap -------------------
    edges = [i / 20 for i in range(21)]
    histogram = []
    for label, attr in (("afdb", "afdb_best_identity"),
                        ("esm_atlas", "esm_atlas_best_identity")):
        values = [getattr(p, attr) for p in proteins]
        values = [v for v in values if v is not None]
        for lo, hi in zip(edges, edges[1:]):
            histogram.append({
                "arm": label, "bin_lo": f"{lo:.2f}", "bin_hi": f"{hi:.2f}",
                "n": sum(1 for v in values if lo <= v < hi),
                "share": f"{sum(1 for v in values if lo <= v < hi) / len(values):.4f}",
            })
        summary[f"{label}_n_with_hit"] = len(values)
        summary[f"{label}_share_in_040_055"] = round(
            sum(1 for v in values if 0.40 <= v < 0.55) / len(values), 4)
        summary[f"{label}_share_above_090"] = round(
            sum(1 for v in values if v >= 0.90) / len(values), 4)
    write(DATA / "arm_identity_histogram.csv", histogram)

    kingdom_rows = []
    for kingdom in sorted({annotation[(p.dataset, p.stem)]["kingdom"]
                           for p in proteins}):
        subset = [p for p in proteins
                  if annotation[(p.dataset, p.stem)]["kingdom"] == kingdom]
        kingdom_rows.append({
            "kingdom": kingdom, "n": len(subset),
            "n_afdb_hit": sum(1 for p in subset if p.afdb_best_identity is not None),
            "n_esm_atlas_hit": sum(
                1 for p in subset if p.esm_atlas_best_identity is not None),
            "share_esm_atlas_hit":
                f"{sum(1 for p in subset if p.esm_atlas_best_identity is not None) / len(subset):.4f}",
            "n_in_eval2": sum(1 for p in subset if p.in_eval2),
            "share_in_eval2": f"{sum(1 for p in subset if p.in_eval2) / len(subset):.4f}",
        })
    write(DATA / "kingdom_by_arm.csv", kingdom_rows)

    # --- A6: does recency predict survival? --------------------------------
    date_rows = []
    for p in proteins:
        ann = annotation[(p.dataset, p.stem)]
        signal = (as_int(ann["is_synthetic_taxon"], 0)
                  or as_int(ann["has_denovo_keyword"], 0))
        if p.designed_any or signal:
            continue  # audited-natural only
        deposit = ann["deposit_date"] or dates.get(p.stem, "")
        if not deposit:
            continue
        date_rows.append({
            "dataset": p.dataset, "stem": p.stem, "deposit_date": deposit,
            "deposit_year": deposit[:4], "in_eval2": int(p.in_eval2),
            "best_identity": "" if p.best_identity is None
                             else f"{p.best_identity:.3f}",
        })
    year_rows = []
    for year in sorted({r["deposit_year"] for r in date_rows}):
        subset = [r for r in date_rows if r["deposit_year"] == year]
        year_rows.append({
            "deposit_year": year, "n_natural": len(subset),
            "n_in_eval2": sum(r["in_eval2"] for r in subset),
            "share_in_eval2": f"{sum(r['in_eval2'] for r in subset) / len(subset):.4f}",
        })
    write(DATA / "survival_by_deposit_year.csv", year_rows)
    write(DATA / "natural_deposit_dates.csv", date_rows)

    (DATA / "analysis_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
