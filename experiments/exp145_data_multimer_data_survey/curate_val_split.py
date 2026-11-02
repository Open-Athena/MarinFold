# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assign confident AFDB dimers to train or validation by subunit homology.

The contacts-v1 monomer models are validated on a fixed AFDB split (exp53,
cluster-consistent on afdb-24M's ``split`` column). A multimer whose subunit is
homologous to one of those validation proteins would leak, so it goes to the
multimer validation set rather than to training.

The rule, per complex: **route to val if EITHER subunit hits a validation monomer
at >= --min-identity sequence identity covering >= --min-coverage of the shorter
of the two sequences.** Either-subunit rather than both is the conservative
reading; it costs training complexes whose partner is a validation protein, and
the summary reports how many so the cost is visible rather than assumed.

On coverage: MMseqs2's ``--cov-mode 5`` is a length-*ratio* filter, not alignment
coverage of the shorter sequence, so it is not the flag this wants. The search
runs permissively and the thresholds are applied here from the reported ``qcov``
and ``tcov``. For a single local alignment the shorter sequence always carries
the higher coverage, so "covers X% of the shorter sequence" is ``max(qcov, tcov)
>= X``. Applying thresholds after the search also means one search can report
several settings, which is why ``--report-also`` exists.

Search cost is asymmetric and small: ~2.0 M query subunits against a ~40 k target
database. Sensitivity is what matters at 30% identity, not speed, hence the
default ``-s 7.5``.

Usage::

    python3 curate_val_split.py --out data/curation
    python3 curate_val_split.py --out data/curation --min-identity 0.5
"""

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# query, target, identity, alignment length, query coverage, target coverage,
# E-value, bit score. Order matters: it is parsed positionally below.
FORMAT_OUTPUT = "query,target,fident,alnlen,qcov,tcov,evalue,bits"

INPUTS = {
    "homodimer": "confident_homodimers.csv.gz",
    "heterodimer": "confident_heterodimers.csv.gz",
}


def find_mmseqs(explicit: str | None) -> str:
    binary = explicit or shutil.which("mmseqs")
    if not binary:
        raise SystemExit(
            "mmseqs not found. Install it (https://github.com/soedinglab/MMseqs2) "
            "or pass --mmseqs /path/to/mmseqs."
        )
    return binary


def concat_shards(shard_dir: Path, target: Path) -> int:
    """Concatenate the subunit shard FASTAs into one query file.

    Returns the record count. Rebuilt whenever the shard set is newer, so adding
    a missing shard and re-running does the right thing.
    """
    shards = sorted(shard_dir.glob("subunits-*.fasta"))
    if not shards:
        raise SystemExit(f"no subunit FASTA shards in {shard_dir}; "
                         "run curate_subunit_sequences.py first")
    newest = max(shard.stat().st_mtime for shard in shards)
    if not target.exists() or target.stat().st_mtime < newest:
        with target.open("wb") as out:
            for shard in shards:
                with shard.open("rb") as handle:
                    shutil.copyfileobj(handle, out)
    with target.open() as handle:
        return sum(1 for line in handle if line.startswith(">"))


def run_search(
    binary: str, queries: Path, targets: Path, out_tsv: Path, tmp: Path,
    sensitivity: float, min_identity: float, threads: int, log: Path,
) -> float:
    """Run ``mmseqs easy-search`` and return elapsed seconds.

    ``--min-seq-id`` is set below the reporting threshold so a stricter setting
    can be evaluated from the same hit table without searching again. ``-c 0``
    leaves coverage entirely to the caller.
    """
    args = [
        binary, "easy-search", str(queries), str(targets), str(out_tsv), str(tmp),
        "--format-output", FORMAT_OUTPUT,
        "-s", str(sensitivity),
        "--min-seq-id", str(max(min_identity - 0.05, 0.0)),
        "-c", "0",
        "--threads", str(threads),
        "--max-seqs", "300",
    ]
    print(f"  $ {' '.join(args)}", flush=True)
    start = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    elapsed = time.time() - start
    log.write_text(proc.stdout + "\n----- stderr -----\n" + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise RuntimeError(f"mmseqs easy-search exited {proc.returncode}:\n{tail}")
    print(f"  ok, {elapsed / 60:.1f} min (log: {log})")
    return elapsed


def load_hits(tsv: Path, min_identity: float, min_coverage: float) -> dict[str, dict[str, Any]]:
    """Map each query accession to its best validation hit passing the thresholds.

    "Best" is by identity, then coverage: the reported hit is the reason the
    complex was routed, so it should be the strongest evidence, not the first
    line MMseqs2 happened to emit.
    """
    best: dict[str, dict[str, Any]] = {}
    with tsv.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            query, target = fields[0], fields[1]
            identity, _alnlen = float(fields[2]), int(fields[3])
            qcov, tcov = float(fields[4]), float(fields[5])
            coverage = max(qcov, tcov)
            if identity < min_identity or coverage < min_coverage:
                continue
            candidate = {
                "val_target": target,
                "identity": round(identity, 4),
                "coverage": round(coverage, 4),
                "evalue": fields[6],
                "bits": fields[7],
            }
            current = best.get(query)
            if current is None or (candidate["identity"], candidate["coverage"]) > (
                current["identity"], current["coverage"]
            ):
                best[query] = candidate
    return best


def assign(
    curation_dir: Path, hits: dict[str, dict[str, Any]], out_path: Path
) -> dict[str, Any]:
    """Write the per-complex split assignment and return the counts."""
    counts = {
        kind: {"train": 0, "val": 0, "val_via_subunit_1": 0, "val_via_subunit_2": 0,
               "val_via_both": 0}
        for kind in INPUTS
    }
    fields = [
        "model_entity_id", "kind", "split", "uniprot_1", "uniprot_2",
        "val_reason", "val_target", "identity", "coverage",
    ]
    with gzip.open(out_path, "wt", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for kind, name in INPUTS.items():
            with gzip.open(curation_dir / name, "rt") as handle:
                for row in csv.DictReader(handle):
                    hit_1 = hits.get(row["uniprot_1"])
                    # A homodimer's two subunits are the same accession; only
                    # count it once so the reason columns stay meaningful.
                    hit_2 = hits.get(row["uniprot_2"]) if kind == "heterodimer" else None
                    chosen = max(
                        (h for h in (hit_1, hit_2) if h),
                        key=lambda h: (h["identity"], h["coverage"]),
                        default=None,
                    )
                    if chosen is None:
                        counts[kind]["train"] += 1
                        writer.writerow({
                            "model_entity_id": row["model_entity_id"], "kind": kind,
                            "split": "train", "uniprot_1": row["uniprot_1"],
                            "uniprot_2": row["uniprot_2"], "val_reason": "",
                            "val_target": "", "identity": "", "coverage": "",
                        })
                        continue
                    counts[kind]["val"] += 1
                    if hit_1 and hit_2:
                        reason = "both"
                        counts[kind]["val_via_both"] += 1
                    elif hit_1:
                        reason = "subunit_1"
                        counts[kind]["val_via_subunit_1"] += 1
                    else:
                        reason = "subunit_2"
                        counts[kind]["val_via_subunit_2"] += 1
                    writer.writerow({
                        "model_entity_id": row["model_entity_id"], "kind": kind,
                        "split": "val", "uniprot_1": row["uniprot_1"],
                        "uniprot_2": row["uniprot_2"], "val_reason": reason,
                        "val_target": chosen["val_target"],
                        "identity": chosen["identity"], "coverage": chosen["coverage"],
                    })
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/curation"))
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--val-fasta", type=Path, default=None)
    parser.add_argument("--min-identity", type=float, default=0.30)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument(
        "--report-also",
        nargs="*",
        type=float,
        default=[0.4, 0.5],
        help="extra identity thresholds to report counts for, from the same search",
    )
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mmseqs", default=None)
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--research", action="store_true",
                        help="force a new mmseqs search even if a hit table exists")
    args = parser.parse_args()

    shard_dir = args.shard_dir or (args.out / "subunit_shards")
    val_fasta = args.val_fasta or (args.out / "val_monomers.fasta")
    if not val_fasta.exists():
        raise SystemExit(f"missing {val_fasta}; run curate_val_reference.py first")

    binary = find_mmseqs(args.mmseqs)
    work = args.out / "mmseqs_work"
    work.mkdir(parents=True, exist_ok=True)

    print("[1/4] assembling query FASTA")
    queries = work / "subunits.fasta"
    n_queries = concat_shards(shard_dir, queries)
    with val_fasta.open() as handle:
        n_targets = sum(1 for line in handle if line.startswith(">"))
    print(f"  {n_queries:,} subunits against {n_targets:,} validation monomers")

    hits_tsv = work / "subunit_vs_val.tsv"
    if hits_tsv.exists() and hits_tsv.stat().st_size > 0 and not args.research:
        # The search is the expensive step (5.0 h for 1.98 M x 42 k at -s 7.5) and
        # thresholds are applied afterwards, so re-deciding the split must not
        # re-run it. --research forces a fresh search when the inputs change.
        print(f"[2/4] reusing existing hit table {hits_tsv} (pass --research to redo)")
        elapsed = 0.0
    else:
        print("[2/4] mmseqs easy-search")
        elapsed = run_search(
            binary, queries, val_fasta, hits_tsv, work / "tmp",
            args.sensitivity, args.min_identity, args.threads, work / "mmseqs.log",
        )

    print("[3/4] applying thresholds")
    hits = load_hits(hits_tsv, args.min_identity, args.min_coverage)
    print(f"  {len(hits):,} subunits hit a validation monomer at "
          f"identity >= {args.min_identity}, coverage >= {args.min_coverage}")

    print("[4/4] assigning complexes")
    tag = f"id{int(round(args.min_identity * 100))}_cov{int(round(args.min_coverage * 100))}"
    assignment = args.out / f"dimer_split_assignment_{tag}.csv.gz"
    counts = assign(args.out, hits, assignment)

    alternates = {}
    for threshold in args.report_also:
        other = load_hits(hits_tsv, threshold, args.min_coverage)
        alternates[f"identity>={threshold}"] = {
            "subunits_hit": len(other),
            "note": "subunit-level count only; rerun with --min-identity to reassign",
        }

    summary: dict[str, Any] = {
        "min_identity": args.min_identity,
        "min_coverage": args.min_coverage,
        "coverage_definition": "max(qcov, tcov), i.e. fraction of the shorter sequence",
        "rule": "complex -> val if EITHER subunit hits a validation monomer",
        "n_query_subunits": n_queries,
        "n_val_targets": n_targets,
        "n_subunits_with_val_hit": len(hits),
        "search_seconds": round(elapsed, 1),
        "sensitivity": args.sensitivity,
        "counts": counts,
        "totals": {
            "train": sum(c["train"] for c in counts.values()),
            "val": sum(c["val"] for c in counts.values()),
        },
        "alternate_thresholds": alternates,
        "assignment": str(assignment),
    }
    (args.out / f"dimer_split_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not args.keep_tmp:
        shutil.rmtree(work / "tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
