# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 — MMseqs2-search all 776 expanded eval proteins against exp199's training set.

Appends :mod:`build_query_set`'s 222 net-new FoldBench monomers to exp213's 554
queries and searches the union against the **existing** 70.9 M-sequence target
database exp213 built. That database is the expensive artifact — 146 GB streamed
and ~80 GB of scratch to rebuild — so it is reused, never regenerated, and every
database this script creates is suffixed ``_expanded`` so exp213's ``queryDB`` /
``alnDB`` / ``aln.m8`` are left untouched and its published numbers stay
reproducible.

This deliberately does **not** call exp213's ``search_overlap.py``: that script
regenerates ``eval_queries.fasta`` from the eval manifests, which would clobber
the expanded FASTA, and it derives the target DB path from its own tag (so a
``_expanded`` run would look for a ``targetDB_expanded`` that does not exist and
rebuild 17 GB from scratch). The *reduction* is imported from it unchanged —
that is the part that has to be bit-identical.

The alignment reduction and the identity conventions come from exp213 verbatim
via :mod:`exp213_link`: a hit counts toward the identity axis only when
``evalue <= 1e-3`` **and** ``qcov >= 0.50``, and the reported identity is the
max ``fident`` over those hits.

    uv run python search_expanded.py --work /data/exp213_overlap
    uv run python search_expanded.py --work /data/exp213_overlap --skip-search
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from build_query_set import DATASET_NEW
from exp213_link import (
    ARMS,
    EXP213_TABLE,
    FORMAT,
    MANIFEST_STRATA,
    check_exp213_queries,
    ensure_mmseqs,
    read_exp213_queries,
    reduce_alignments,
    run,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: exp213's *actual* search parameters, read off its run log
#: (``/data/exp213_overlap/search_full.log``) rather than its argparse defaults:
#: the full two-arm search ran with ``--max-seqs 5000``, which is what its
#: committed table and the 284/264 anchor were computed under. Its
#: ``eval_train_identity.provenance.json`` agrees. (The AFDB-only cross-check ran
#: at the 2000 default; issue #226's quoted command line copies that number, but
#: matching the *published* table is what parity means here.)
MAX_SEQS = 5000
SENSITIVITY = 7.5
REPORTING_EVALUE = 10.0

#: Suffix on every database this script writes, so nothing exp213 published can
#: be overwritten by a re-run.
TAG = "expanded"


def build_expanded_fasta(rest_fasta: Path, out: Path) -> tuple[int, int]:
    """exp213's 554 records verbatim, then the net-new ones. Returns the counts.

    Order matters only for reproducibility, but *content* matters a great deal:
    the first 554 records must be exp213's file unchanged, or the parity check
    against its 284/264 is comparing two different query sets.
    """
    check_exp213_queries()
    existing = read_exp213_queries()
    rest = [
        (h, s) for h, s in _read_fasta(rest_fasta)
    ]
    headers = [h for h, _ in existing] + [h for h, _ in rest]
    if len(set(headers)) != len(headers):
        duplicates = sorted({h for h in headers if headers.count(h) > 1})
        raise SystemExit(f"duplicate query headers in the expanded set: {duplicates}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f">{h}\n{s}\n" for h, s in existing + rest))
    return len(existing), len(rest)


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def search(work: Path, query_fasta: Path, *, threads: int,
           split_memory_limit: str) -> Path:
    """Search the expanded queries against exp213's existing ``targetDB``."""
    target_db = work / "targetDB"
    if not target_db.exists():
        raise SystemExit(
            f"{target_db} is gone. Rebuilding it is ~146 GB streamed and ~80 GB of "
            "scratch — stop and re-scope per issue #226 (fold it into #225's Stage 2)."
        )
    mmseqs = ensure_mmseqs()
    print(f"[mmseqs] binary: {mmseqs}", flush=True)
    query_db = work / f"queryDB_{TAG}"
    aln_db = work / f"alnDB_{TAG}"
    tmp = work / f"mmseqs_tmp_{TAG}"

    # mmseqs refuses to overwrite an existing result DB, so a re-run must clear
    # the query/aln databases first. The target DB is never touched.
    for stale in list(work.glob(f"alnDB_{TAG}*")) + list(work.glob(f"queryDB_{TAG}*")):
        stale.unlink()

    run([mmseqs, "createdb", query_fasta, query_db])
    t0 = time.time()
    run([mmseqs, "search", query_db, target_db, aln_db, tmp,
         "-s", SENSITIVITY, "--max-seqs", MAX_SEQS, "-e", REPORTING_EVALUE,
         "--threads", threads, "--split-memory-limit", split_memory_limit])
    print(f"[mmseqs] search in {time.time() - t0:.0f}s", flush=True)

    m8 = work / f"aln_{TAG}.m8"
    run([mmseqs, "convertalis", query_db, target_db, aln_db, m8,
         "--format-output", FORMAT, "--threads", threads])
    print(f"[mmseqs] alignments -> {m8}", flush=True)
    return m8


def build_meta(rest_fasta: Path) -> dict[str, dict]:
    """``{dataset}__{stem}`` -> the strata columns exp213's reducer expects.

    The 554 carry exp213's committed values through unchanged, so the expanded
    table joins onto its predecessor on ``(dataset, stem)`` with identical
    metadata. The 222 net-new have no Foldseek verdict and no MSA Neff — those
    axes are separate compute that #226 does not run — so their strata columns
    are empty and only ``length`` is filled.
    """
    meta: dict[str, dict] = {}
    with EXP213_TABLE.open() as fh:
        for row in csv.DictReader(fh):
            key = f"{row['dataset']}__{row['stem']}"
            meta[key] = {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "query_len": int(row["query_len"]),
                **{c: row.get(c, "") for c in MANIFEST_STRATA},
            }
    for header, sequence in _read_fasta(rest_fasta):
        dataset, stem = header.split("__", 1)
        if dataset != DATASET_NEW:
            raise SystemExit(f"unexpected dataset {dataset!r} in {rest_fasta}")
        meta[header] = {
            "dataset": dataset,
            "stem": stem,
            "query_len": len(sequence),
            **{c: "" for c in MANIFEST_STRATA},
            "length": len(sequence),
        }
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path("/data/exp213_overlap"),
                    help="exp213's work dir; must still hold its targetDB")
    ap.add_argument("--rest-fasta", type=Path,
                    default=DATA / "foldbench_rest_queries.fasta")
    ap.add_argument("--query-fasta", type=Path, default=DATA / "eval_queries_expanded.fasta")
    ap.add_argument("--out", type=Path, default=DATA / "eval_train_identity_expanded.csv")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--split-memory-limit", default="200G")
    ap.add_argument("--skip-search", action="store_true",
                    help="reuse an existing aln_expanded.m8 and only rebuild the table")
    args = ap.parse_args()

    n_existing, n_new = build_expanded_fasta(args.rest_fasta, args.query_fasta)
    print(f"[queries] {n_existing} exp213 + {n_new} net-new = {n_existing + n_new} "
          f"-> {args.query_fasta}", flush=True)

    m8 = args.work / f"aln_{TAG}.m8"
    if not args.skip_search:
        m8 = search(args.work, args.query_fasta, threads=args.threads,
                    split_memory_limit=args.split_memory_limit)
    elif not m8.exists():
        raise SystemExit(f"--skip-search but {m8} does not exist")

    meta = build_meta(args.rest_fasta)
    rows = reduce_alignments(m8, meta, MAX_SEQS)
    if len(rows) != n_existing + n_new:
        raise SystemExit(f"expected {n_existing + n_new} rows, reduced to {len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table] {len(rows)} rows -> {args.out}", flush=True)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["stratum"]] += 1
    print("[strata]", dict(counts), flush=True)
    for arm in ARMS:
        n = sum(1 for r in rows if r[f"{arm}_n_hits_significant"] > 0)
        print(f"[{arm}] {n}/{len(rows)} eval proteins have a significant hit", flush=True)
    censored = sum(r["hits_censored"] for r in rows)
    if censored:
        print(f"[warn] {censored} proteins hit the --max-seqs {MAX_SEQS} cap; "
              "their n_hits columns are censored (best-hit columns are not)", flush=True)

    args.out.with_suffix(".provenance.json").write_text(json.dumps({
        "sensitivity": SENSITIVITY, "max_seqs": MAX_SEQS,
        "reporting_evalue": REPORTING_EVALUE, "mmseqs": ensure_mmseqs(),
        "target_db": str(args.work / "targetDB"),
        "target_db_from": "exp213 (issue #213); 70,889,604 training sequences",
        "n_queries": len(rows),
        "n_queries_from_exp213": n_existing,
        "n_queries_net_new": n_new,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
