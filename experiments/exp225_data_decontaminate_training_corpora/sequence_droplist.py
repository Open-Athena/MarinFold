# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 2a — the sequence-axis drop list: which training rows are contaminated.

#213 asked "which *eval* proteins have a training homolog" and so kept one row
per query. This asks the inverted question — "which *training* rows are a
homolog of some eval protein" — and therefore has to keep **every** hit. That
is the only real difference, and it is why the search is re-run instead of
reduced from ``aln.m8``: #213's run capped the prefilter at ``--max-seqs 2000``
and 96 of the 554 queries hit the cap, so its alignments are a censored sample
of exactly the thing this list needs to be complete about.

Everything expensive is reused. ``/data/exp213_overlap/targetDB`` is the
70,889,604-sequence MMseqs2 database over both training corpora, and every
target in it is headed ``{arm}|{shard:05d}_{row}_{entry_id}``. So a hit names
the corpus, the shard and the row it came from, and inverting hits into a drop
list needs no join against the corpus at all — decontamination is a row filter
on ``entry_id``, exactly as the issue argues.

**Threshold.** Tier A drops a training row when any alignment to any of the 554
satisfies ``identity >= 30% over >= 50% query coverage`` **or** ``E <= 1e-3``
(:func:`decontam_lib.is_sequence_contaminant`). The identity bar is exp65's
``REDUNDANT_ID``, tighter than the 40 % #91's ESM-Atlas funnel used; the E-value
bar is #213's significance line, and catches remote homologs that align over
too little of the query to clear the coverage gate.

**The reporting threshold is load-bearing, and is therefore exp65's.** The
E-value arm is self-limiting, but the identity arm is not: it asks only for
30 % identity over half the query, and for a *short* query that is reachable by
chance, so how many rows it drops depends on how deep mmseqs is asked to
report. exp65's ``seq_leakage.py`` — the source of the 30 % bar in the first
place — searched at ``-e 10``, and #213 used the same, so ``E <= 10`` is what
makes this list's drops mean what ``redundant_seq`` already means elsewhere in
the repo. The search runs far looser than that (``--search-evalue``) and the
tier's ceiling is applied at reduce time (``--report-evalue-ceiling``), so
``sweep_evalue.py`` can price the whole curve off one search and the choice is
measured rather than asserted.

    uv run python sequence_droplist.py --work /data/exp225_decontam
    uv run python sequence_droplist.py --work /data/exp225_decontam --skip-search
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from decontam_lib import (
    ARMS,
    CORPORA,
    REFERENCE_VERSION,
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    TIER_A,
    ensure_mmseqs,
    is_sequence_contaminant,
    parse_target,
    run,
)

HERE = Path(__file__).resolve().parent

#: exp213's target database: both corpora, 70,889,604 sequences, 17 GB. Built
#: once by its ``fetch_train_sequences.py`` + ``concat_targets``; rebuilding it
#: is a 146 GB stream and several hours, so this experiment consumes it.
DEFAULT_TARGET_DB = Path("/data/exp213_overlap/targetDB")

#: Same field list as exp213, so the two runs' alignments are comparable
#: line-for-line.
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits"
FIELDS = FORMAT.split(",")

DROPLIST_SCHEMA = pa.schema(
    [
        ("arm", pa.string()),
        ("entry_id", pa.string()),
        ("shard", pa.int32()),
        ("row", pa.int32()),
        ("best_identity", pa.float32()),
        ("best_qcov", pa.float32()),
        ("best_evalue", pa.float64()),
        ("nearest_eval_key", pa.string()),
        ("rule", pa.string()),
    ]
)

#: Which arm of the Tier A disjunction fired. Recorded per row because the
#: split is the diagnostic that says whether tightening #91's 40 % bar to 30 %
#: actually bought anything.
RULE_EVALUE = "evalue"
RULE_IDENTITY = "identity_coverage"
RULE_BOTH = "both"


def search(
    work: Path,
    queries: Path,
    target_db: Path,
    *,
    sensitivity: float,
    max_seqs: int,
    evalue: float,
    threads: int,
    split_memory_limit: str,
) -> Path:
    """Search the 554 against the training DB keeping every hit; return the ``.m8``."""
    mmseqs = ensure_mmseqs()
    print(f"[mmseqs] binary: {mmseqs}", flush=True)
    query_db = work / "queryDB"
    aln_db = work / "alnDB"
    tmp = work / "mmseqs_tmp"

    # mmseqs refuses to overwrite an existing result DB or search tmp.
    shutil.rmtree(tmp, ignore_errors=True)
    for stale in list(work.glob("alnDB*")) + list(work.glob("queryDB*")):
        stale.unlink()

    run([mmseqs, "createdb", queries, query_db])
    t0 = time.time()
    run(
        [
            mmseqs, "search", query_db, target_db, aln_db, tmp,
            "-s", sensitivity,
            "--max-seqs", max_seqs,
            "-e", evalue,
            "--threads", threads,
            "--split-memory-limit", split_memory_limit,
        ]
    )
    print(f"[mmseqs] search in {time.time() - t0:.0f}s", flush=True)

    m8 = work / "aln_all_hits.m8"
    t0 = time.time()
    run(
        [mmseqs, "convertalis", query_db, target_db, aln_db, m8,
         "--format-output", FORMAT, "--threads", threads]
    )
    print(
        f"[mmseqs] alignments -> {m8} ({m8.stat().st_size / 1e9:.1f} GB, "
        f"{time.time() - t0:.0f}s)",
        flush=True,
    )
    return m8


def build_droplist(m8: Path, report_ceiling: float) -> tuple[dict[tuple[str, str], dict], dict]:
    """Fold every alignment into one record per contaminated training row.

    Keyed on ``(arm, entry_id)`` because ``entry_id`` is the column the
    published parquet carries and therefore what a filter can actually match
    on. ``(shard, row)`` rides along for provenance and to make the filter
    checkable shard by shard.

    ``report_ceiling`` is the reporting threshold *applied at reduce time*,
    deliberately decoupled from the one the search ran at. Searching once at a
    very loose threshold and reducing at a tighter one means the whole
    sensitivity curve (``sweep_evalue.py``) comes out of a single search
    instead of one search per point.
    """
    dropped: dict[tuple[str, str], dict] = {}
    # Two counters on purpose. ``per_query_alignments`` counts what mmseqs
    # actually reported and is what the --max-seqs censoring check has to be
    # made against; ``per_query_hits`` counts only what survives the tier's
    # reporting ceiling. Conflating them would make a search that *was*
    # censored look uncensored whenever the ceiling is tighter than the search.
    per_query_alignments: Counter[str] = Counter()
    per_query_hits: Counter[str] = Counter()
    per_query_drops: defaultdict[str, set] = defaultdict(set)
    n_alignments = 0
    identity_only_above_evalue = 0
    t0 = time.time()

    with m8.open() as fh:
        for line in fh:
            values = line.rstrip("\n").split("\t")
            hit = dict(zip(FIELDS, values))
            identity, qcov = float(hit["fident"]), float(hit["qcov"])
            evalue = float(hit["evalue"])
            query = hit["query"]
            per_query_alignments[query] += 1
            if evalue > report_ceiling:
                continue

            per_query_hits[query] += 1
            n_alignments += 1
            if n_alignments % 20_000_000 == 0:
                print(
                    f"[reduce] {n_alignments:,} alignments, {len(dropped):,} rows dropped, "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

            if not is_sequence_contaminant(identity, qcov, evalue):
                continue

            by_evalue = evalue <= SEQ_MAX_EVALUE
            by_identity = identity >= SEQ_MIN_IDENTITY and qcov >= SEQ_MIN_QCOV
            if by_identity and not by_evalue:
                identity_only_above_evalue += 1
            rule = RULE_BOTH if (by_evalue and by_identity) else (
                RULE_EVALUE if by_evalue else RULE_IDENTITY
            )

            target = parse_target(hit["target"])
            key = (target.arm, target.entry_id)
            per_query_drops[query].add(key)
            current = dropped.get(key)
            # "Nearest" is by identity, not bitscore: the drop list's job is to
            # justify a removal, and identity is the axis the tier is stated in.
            if current is None or identity > current["best_identity"]:
                dropped[key] = {
                    "arm": target.arm,
                    "entry_id": target.entry_id,
                    "shard": target.shard,
                    "row": target.row,
                    "best_identity": identity,
                    "best_qcov": qcov,
                    "best_evalue": evalue,
                    "nearest_eval_key": query,
                    "rule": rule,
                }
            elif current["rule"] != rule:
                current["rule"] = RULE_BOTH

    stats = {
        "n_alignments": n_alignments,
        "n_alignments_reported": sum(per_query_alignments.values()),
        "n_dropped_rows": len(dropped),
        "identity_arm_hits_above_evalue_bar": identity_only_above_evalue,
        "per_query_alignments": dict(per_query_alignments),
        "per_query_hits": dict(per_query_hits),
        "per_query_drops": {q: len(v) for q, v in per_query_drops.items()},
        "reduce_seconds": round(time.time() - t0, 1),
    }
    return dropped, stats


def write_droplist(dropped: dict[tuple[str, str], dict], out: Path) -> None:
    """One parquet, sorted by ``(arm, shard, row)`` so it reads shard-aligned."""
    out.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(dropped.values(), key=lambda r: (r["arm"], r["shard"], r["row"]))
    columns = {
        name: [record[name] for record in records] for name in DROPLIST_SCHEMA.names
    }
    pq.write_table(pa.table(columns, schema=DROPLIST_SCHEMA), out, compression="zstd")
    print(f"[droplist] {len(records):,} rows -> {out} "
          f"({out.stat().st_size / 1e6:.0f} MB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--target-db", type=Path, default=DEFAULT_TARGET_DB)
    ap.add_argument("--queries", type=Path, default=HERE / "data/reference/eval_queries.fasta")
    ap.add_argument("-s", "--sensitivity", type=float, default=7.5,
                    help="exp65/exp94/exp213 all used 7.5; keep it so the numbers compare")
    ap.add_argument("--max-seqs", type=int, default=1_000_000,
                    help="prefilter hits kept per query. #213 used 2000 and censored 96 "
                         "queries; the run reports the observed maximum so censoring here "
                         "is visible rather than assumed away")
    ap.add_argument("--search-evalue", type=float, default=1000.0,
                    help="how deep mmseqs is asked to report. Deliberately far looser than "
                         "the tier's own threshold so the reduce below — and sweep_evalue.py "
                         "— can be run at any tighter ceiling without a second search")
    ap.add_argument("--report-evalue-ceiling", type=float, default=10.0,
                    help="alignments above this are ignored at reduce time. 10 is exp65's "
                         "and #213's, and is what makes a drop here mean what redundant_seq "
                         "means elsewhere in the repo")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--split-memory-limit", default="200G")
    ap.add_argument("--skip-search", action="store_true",
                    help="reuse an existing aln_all_hits.m8 and only rebuild the list")
    ap.add_argument("--droplist-out", type=Path, default=None,
                    help="default: <work>/droplist_sequence.parquet")
    ap.add_argument("--summary-out", type=Path,
                    default=HERE / "data/sequence_droplist_summary.csv")
    ap.add_argument("--per-query-out", type=Path,
                    default=HERE / "data/sequence_droplist_per_query.csv")
    ap.add_argument("--provenance-out", type=Path,
                    default=HERE / "data/sequence_droplist.provenance.json")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    droplist_out = args.droplist_out or args.work / "droplist_sequence.parquet"
    m8 = args.work / "aln_all_hits.m8"
    if not args.skip_search:
        m8 = search(
            args.work, args.queries, args.target_db,
            sensitivity=args.sensitivity, max_seqs=args.max_seqs, evalue=args.search_evalue,
            threads=args.threads, split_memory_limit=args.split_memory_limit,
        )
    elif not m8.exists():
        raise SystemExit(f"--skip-search but {m8} does not exist")

    dropped, stats = build_droplist(m8, args.report_evalue_ceiling)
    write_droplist(dropped, droplist_out)

    per_arm = Counter(record["arm"] for record in dropped.values())
    per_rule = Counter(record["rule"] for record in dropped.values())
    rows = []
    for arm in ARMS:
        corpus = CORPORA[arm]
        n_dropped = per_arm[arm]
        rows.append(
            {
                "arm": arm,
                "label": corpus.label,
                "n_documents": corpus.n_documents,
                "n_dropped": n_dropped,
                "pct_dropped": round(100 * n_dropped / corpus.n_documents, 4),
                "n_surviving": corpus.n_documents - n_dropped,
                "tier": TIER_A,
                "reference_version": REFERENCE_VERSION,
            }
        )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"[survival] {row['arm']}: dropped {row['n_dropped']:,} / "
              f"{row['n_documents']:,} ({row['pct_dropped']:.3f}%)", flush=True)

    # Per eval protein: how much of the training corpus it is responsible for
    # removing. Small enough to commit, and it is what identifies the handful
    # of eval proteins that dominate the cost of Tier A.
    queries = [line[1:].strip() for line in args.queries.read_text().splitlines()
               if line.startswith(">")]
    with args.per_query_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "dataset", "stem", "n_alignments", "n_rows_dropped"])
        for key in queries:
            dataset, _, stem = key.partition("__")
            writer.writerow([key, dataset, stem,
                             stats["per_query_alignments"].get(key, 0),
                             stats["per_query_drops"].get(key, 0)])
    print(f"[per-query] {len(queries)} rows -> {args.per_query_out}", flush=True)

    max_hits = max(stats["per_query_alignments"].values(), default=0)
    censored = [q for q, n in stats["per_query_alignments"].items() if n >= args.max_seqs]
    if censored:
        print(f"[warn] {len(censored)} queries reached --max-seqs {args.max_seqs:,}; "
              "the drop list is INCOMPLETE for them — re-run higher", flush=True)
    else:
        print(f"[check] no censoring: busiest query had {max_hits:,} alignments "
              f"against a --max-seqs of {args.max_seqs:,}", flush=True)
    n_identity_arm = per_rule[RULE_IDENTITY]
    print(f"[rules] {per_rule[RULE_EVALUE]:,} rows by E-value alone, "
          f"{n_identity_arm:,} by identity+coverage alone, {per_rule[RULE_BOTH]:,} by both. "
          f"The identity-alone rows are the ones the E <= {args.report_evalue_ceiling:g} "
          "reporting ceiling can move — see sweep_evalue.py", flush=True)

    args.provenance_out.write_text(
        json.dumps(
            {
                "reference_version": REFERENCE_VERSION,
                "tier": TIER_A,
                "target_db": str(args.target_db),
                "sensitivity": args.sensitivity,
                "max_seqs": args.max_seqs,
                "search_evalue": args.search_evalue,
                "report_evalue_ceiling": args.report_evalue_ceiling,
                "thresholds": {
                    "min_identity": SEQ_MIN_IDENTITY,
                    "min_qcov": SEQ_MIN_QCOV,
                    "max_evalue": SEQ_MAX_EVALUE,
                },
                "n_alignments": stats["n_alignments"],
                "n_dropped_rows": stats["n_dropped_rows"],
                "dropped_by_arm": dict(per_arm),
                "dropped_by_rule": dict(per_rule),
                "identity_arm_hits_above_evalue_bar":
                    stats["identity_arm_hits_above_evalue_bar"],
                "max_alignments_for_one_query": max_hits,
                "censored_queries": censored,
                "droplist": str(droplist_out),
                "reduce_seconds": stats["reduce_seconds"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[provenance] -> {args.provenance_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
