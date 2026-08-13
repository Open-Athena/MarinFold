# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 -- audit the generated corpora and draw the summary plots.

Answers the questions a reader of the experiment will actually ask:

* **Where did 195,858 entries go?** The funnel, reconstructed from the
  entry-level counts plus the per-entry ledger, with every drop attributed to
  a named filter.
* **What is in the corpora?** Document counts, token totals, length and
  chain-count distributions, truncation rate, and -- for the multimers -- how
  much of each document is interface rather than intra-chain contact.
* **Did the eval set leak in?** Exact-sequence overlap between the corpora and
  the 554-chain contact eval set, on top of the PDB-id exclusion already
  applied during curation.

Writes CSVs to ``data/`` (committed, so plots regenerate without the corpus)
and PNGs to ``plots/``.

Usage::

    uv run python qc.py --root /data/exp222_pdb_curation
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds

from build_summary import save_plot_with_meta



_DOC_COLUMNS = [
    "entry_id", "pdb_id", "subset", "seq_len", "num_tokens", "num_chains",
    "chain_lengths", "contacts_emitted", "contacts_pre_filter",
    "contacts_emitted_inter_chain", "contacts_pre_filter_inter_chain",
    "truncated", "release_date", "resolution", "method", "cluster_ids",
    "resolved_seq_sha1",
]


def load_docs(root: Path, subset: str):
    directory = root / "docs" / subset
    if not directory.is_dir() or not any(directory.glob("*.parquet")):
        return None
    return ds.dataset(directory, format="parquet").to_table(columns=_DOC_COLUMNS)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def funnel(root: Path, out_dir: Path) -> None:
    """Reconstruct the entry -> document funnel from the ledger."""
    ledger_dir = root / "ledger"
    if not ledger_dir.is_dir():
        print("no ledger; skipping funnel")
        return
    table = ds.dataset(ledger_dir, format="parquet").to_table()
    n_entries = table.num_rows
    errors = sum(1 for e in table.column("error").to_pylist() if e)
    multimer_status = Counter(table.column("multimer_status").to_pylist())

    asu_drop_reasons: Counter = Counter()
    for reasons in table.column("asu_drop_reasons").to_pylist():
        asu_drop_reasons.update(reasons or [])
    assembly_drop_reasons: Counter = Counter()
    for reasons in table.column("assembly_drop_reasons").to_pylist():
        assembly_drop_reasons.update(reasons or [])

    rows = [
        {"stage": "entries processed", "count": n_entries},
        {"stage": "entries with an error", "count": errors},
        {"stage": "ASU protein chains kept",
         "count": sum(table.column("asu_chains_kept").to_pylist())},
        {"stage": "ASU protein chains dropped",
         "count": sum(table.column("asu_chains_dropped").to_pylist())},
        {"stage": "monomer documents",
         "count": sum(table.column("monomer_docs").to_pylist())},
    ]
    # One ledger entry per dropped chain, so these sum to the total above.
    rows += [
        {"stage": f"ASU chain drop: {reason}", "count": count}
        for reason, count in asu_drop_reasons.most_common()
    ]
    rows += [
        {"stage": f"assembly chain drop: {reason}", "count": count}
        for reason, count in assembly_drop_reasons.most_common()
    ]
    rows += [
        {"stage": f"multimer outcome: {status}", "count": count}
        for status, count in multimer_status.most_common()
    ]
    write_csv(out_dir / "funnel.csv", rows)

    # Chains that failed to serialize despite passing curation -- these are the
    # ones worth eyeballing, since curation thought they were fine.
    failures: Counter = Counter()
    for entries in table.column("monomer_failures").to_pylist():
        for entry in entries or []:
            failures[entry.split(": ", 1)[-1][:80]] += 1
    write_csv(
        out_dir / "monomer_failures.csv",
        [{"reason": reason, "count": count} for reason, count in failures.most_common(50)],
    )


def corpus_stats(tables: dict, out_dir: Path) -> None:
    rows = []
    for subset, table in tables.items():
        if table is None:
            continue
        tokens = np.asarray(table.column("num_tokens").to_pylist())
        seq_len = np.asarray(table.column("seq_len").to_pylist())
        chains = np.asarray(table.column("num_chains").to_pylist())
        emitted = np.asarray(table.column("contacts_emitted").to_pylist())
        inter = np.asarray(table.column("contacts_emitted_inter_chain").to_pylist())
        truncated = np.asarray(table.column("truncated").to_pylist())
        seq_hashes = table.column("resolved_seq_sha1").to_pylist()
        clusters = {
            c for ids in table.column("cluster_ids").to_pylist()
            for c in (ids or []) if c >= 0
        }
        rows.append({
            "subset": subset,
            "documents": len(tokens),
            "distinct_pdb_entries": len(set(table.column("pdb_id").to_pylist())),
            "distinct_resolved_sequences": len(set(seq_hashes)),
            "distinct_clusters_40pct": len(clusters),
            "total_tokens": int(tokens.sum()),
            "mean_tokens": round(float(tokens.mean()), 1),
            "median_seq_len": int(np.median(seq_len)),
            "mean_seq_len": round(float(seq_len.mean()), 1),
            "max_seq_len": int(seq_len.max()),
            "mean_chains": round(float(chains.mean()), 3),
            "max_chains": int(chains.max()),
            "mean_contacts": round(float(emitted.mean()), 1),
            "total_contacts": int(emitted.sum()),
            "inter_chain_contacts": int(inter.sum()),
            "inter_chain_fraction": round(
                float(inter.sum() / emitted.sum()) if emitted.sum() else 0.0, 4
            ),
            "truncated_fraction": round(float(truncated.mean()), 4),
        })
    write_csv(out_dir / "corpus_stats.csv", rows)
    for row in rows:
        print(row)


def leakage_audit(
    tables: dict, eval_csv: Path, clusters_path: Path, out_dir: Path
) -> None:
    """How close the corpora get to the contact eval set, two ways.

    Curation already removed every eval PDB **entry** by id. What remains is
    the question exp213 asked of the AFDB corpus: how much of the training set
    is *homologous* to the eval set?

    1. **Exact sequence** -- a different entry depositing the identical
       resolved sequence. Both sides are hashed in the same 3-letter residue
       space (the eval manifest's one-letter ``input_seq`` is mapped through
       contacts-v1's own converter). This is a strict lower bound: the eval
       sequence is the *input* sequence, so a corpus chain with unresolved
       residues will not match even when it is the same protein.
    2. **40% cluster**, reported from both ends, because the two answer
       different questions and only one is comparable to exp213:

       - ``cluster40_fraction`` -- what share of the *corpus* is homologous to
         the eval set. Small by construction (the eval set is 554 chains and
         the corpus is hundreds of thousands), and it says how much of the
         training data is suspect.
       - ``eval_chains_covered_fraction`` -- what share of the *eval set* has a
         homolog somewhere in the corpus. This is the exp213 number: that
         experiment found the eval set 58% homologous to exp199's AFDB
         training data. It says how much of the benchmark is answerable from
         memory.

       Both credit every entity of an eval entry rather than only the chain
       actually evaluated, so both are upper bounds.

    Neither is a filter. exp213 found the eval score is *not* homology-inflated
    (rho ~0 against sequence identity), so the useful output is the measured
    number, not a purge.
    """
    from curate import load_clusters
    from marinfold.document_structures.contacts_v1 import residues_from_sequence

    eval_pdb_ids: set[str] = set()
    with eval_csv.open() as handle:
        for row in csv.DictReader(handle):
            eval_pdb_ids.add(row["pdb_id"].strip().lower())

    # The eval manifests carry the input sequences; hash them the way the
    # corpus hashes its resolved residues.
    eval_hashes: dict[str, str] = {}
    manifests = sorted(
        Path("../exp74_evals_protenix_pyconfind_contacts/data").glob("eval_manifest_*.csv")
    )
    for manifest in manifests:
        with manifest.open() as handle:
            for row in csv.DictReader(handle):
                sequence = "".join(row.get("input_seq", "").split()).upper()
                if not sequence:
                    continue
                three = "".join(r.resname for r in residues_from_sequence(sequence))
                eval_hashes.setdefault(
                    hashlib.sha1(three.encode()).hexdigest(), row["stem"]
                )

    clusters = load_clusters(str(clusters_path))
    # cluster id -> the eval PDB ids that land in it, so coverage can be
    # counted from the eval side as well as the corpus side.
    eval_clusters: dict[int, set[str]] = {}
    for key, cluster_id in clusters.items():
        pdb_id = key.split("_")[0].lower()
        if pdb_id in eval_pdb_ids:
            eval_clusters.setdefault(cluster_id, set()).add(pdb_id)
    # An eval entry with no entity in the cluster file cannot be matched at
    # all; count it so the denominator is honest.
    eval_ids_clustered = {pid for ids in eval_clusters.values() for pid in ids}

    rows = []
    for subset, table in tables.items():
        if table is None:
            continue
        entry_ids = table.column("entry_id").to_pylist()
        pdb_ids = table.column("pdb_id").to_pylist()
        digests = table.column("resolved_seq_sha1").to_pylist()
        cluster_lists = table.column("cluster_ids").to_pylist()

        exact = 0
        examples: list[str] = []
        for entry_id, digest in zip(entry_ids, digests):
            stem = eval_hashes.get(digest)
            if stem is None:
                continue
            exact += 1
            if len(examples) < 10:
                examples.append(f"{entry_id}->{stem}")
        homologous = 0
        covered_eval_ids: set[str] = set()
        for ids in cluster_lists:
            hit = [eval_clusters[c] for c in (ids or []) if c in eval_clusters]
            if hit:
                homologous += 1
                for group in hit:
                    covered_eval_ids |= group
        rows.append({
            "subset": subset,
            "documents": table.num_rows,
            "eval_pdb_ids_present": len(set(pdb_ids) & eval_pdb_ids),
            "exact_sequence_matches": exact,
            "exact_sequence_fraction": round(exact / table.num_rows, 6) if table.num_rows else 0.0,
            "cluster40_homologous": homologous,
            "cluster40_fraction": round(homologous / table.num_rows, 6) if table.num_rows else 0.0,
            "eval_entries_total": len(eval_pdb_ids),
            "eval_entries_clustered": len(eval_ids_clustered),
            "eval_entries_covered": len(covered_eval_ids),
            "eval_entries_covered_fraction": round(
                len(covered_eval_ids) / len(eval_pdb_ids), 4
            ) if eval_pdb_ids else 0.0,
            "examples": "; ".join(examples),
        })
    write_csv(out_dir / "leakage_audit.csv", rows)
    for row in rows:
        print({k: v for k, v in row.items() if k != "examples"})


def plots(tables: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # The deduped corpus is a subset of these two, so it gets statistics and a
    # leakage row but no separate figure -- its distributions are the same
    # shapes with the redundancy taken out.
    monomers, multimers = tables.get("monomers"), tables.get("multimers")

    if monomers is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(monomers.column("seq_len").to_pylist(), bins=60, color="#4C78A8")
        axes[0].set_xlabel("residues"); axes[0].set_ylabel("chains")
        axes[0].set_title("PDB monomer chain length")
        axes[1].hist(monomers.column("num_tokens").to_pylist(), bins=60, color="#4C78A8")
        axes[1].set_xlabel("tokens"); axes[1].set_ylabel("documents")
        axes[1].set_title("PDB monomer document length")
        fig.tight_layout()
        save_plot_with_meta(
            fig, str(plots_dir / "monomer_lengths.png"),
            caption="Residue count and token count of the PDB monomer corpus. "
                    "The 8192-token ceiling is what truncates the right tail.",
        )
        plt.close(fig)

    if multimers is not None:
        chains = np.asarray(multimers.column("num_chains").to_pylist())
        emitted = np.asarray(multimers.column("contacts_emitted").to_pylist())
        inter = np.asarray(multimers.column("contacts_emitted_inter_chain").to_pylist())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(chains, bins=np.arange(1.5, min(chains.max(), 30) + 1.5),
                     color="#F58518")
        axes[0].set_xlabel("protein chains in assembly 1")
        axes[0].set_ylabel("documents")
        axes[0].set_title("Multimer chain count")
        fraction = np.divide(inter, emitted, out=np.zeros_like(inter, dtype=float),
                             where=emitted > 0)
        axes[1].hist(fraction, bins=50, color="#F58518")
        axes[1].set_xlabel("fraction of emitted contacts that cross a chain boundary")
        axes[1].set_ylabel("documents")
        axes[1].set_title("Interface content of a multimer document")
        fig.tight_layout()
        save_plot_with_meta(
            fig, str(plots_dir / "multimer_chains_and_interface.png"),
            caption="How many chains a multimer document carries, and how much "
                    "of its contact budget is spent on interfaces rather than "
                    "intra-chain structure.",
        )
        plt.close(fig)

    if monomers is not None:
        dates = [d[:4] for d in monomers.column("release_date").to_pylist() if d]
        by_year = Counter(dates)
        years = sorted(by_year)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(years, [by_year[y] for y in years], color="#54A24B")
        axes[0].set_xlabel("release year"); axes[0].set_ylabel("chains")
        axes[0].set_title("Release date (cutoff 2021-09-30)")
        axes[0].tick_params(axis="x", rotation=90, labelsize=6)
        resolutions = [r for r in monomers.column("resolution").to_pylist() if r]
        axes[1].hist(resolutions, bins=60, color="#54A24B")
        axes[1].set_xlabel("resolution (A)"); axes[1].set_ylabel("chains")
        axes[1].set_title("Reported resolution (cutoff 9 A)")
        fig.tight_layout()
        save_plot_with_meta(
            fig, str(plots_dir / "provenance.png"),
            caption="Release-year and resolution composition of the monomer "
                    "corpus, the two entry-level Protenix filters.",
        )
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/exp222_pdb_curation"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--eval-csv", type=Path, default=Path("data/eval_set_pdb_ids.csv"))
    parser.add_argument(
        "--clusters", type=Path,
        default=Path("/data/exp222_pdb_curation/metadata/clusters-by-entity-40.txt"),
    )
    args = parser.parse_args(argv)

    tables = {
        "monomers": load_docs(args.root, "monomers"),
        "multimers": load_docs(args.root, "multimers"),
        "deduped": load_docs(args.root, "deduped"),
    }
    funnel(args.root, args.data_dir)
    corpus_stats(tables, args.data_dir)
    leakage_audit(tables, args.eval_csv, args.clusters, args.data_dir)
    plots(tables, args.plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
