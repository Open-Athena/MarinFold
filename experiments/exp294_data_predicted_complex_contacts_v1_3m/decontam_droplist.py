# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A3 — the accession-level evaluation-decontamination drop list.

#225 asks "which *training rows* are a homolog of some eval protein" and keeps
**every** hit rather than one per query. This asks the same question of AFCDB's
subunits, so the rule, the search depth and the reporting ceiling are #225's,
imported from ``decontam_lib`` rather than restated:

* drop when ``E <= 1e-3`` **or** ``identity >= 30% over >= 50% query coverage``
  (:func:`decontam_lib.is_sequence_contaminant` — a disjunction, not a
  conjunct, so remote homologs that align over too little of the query still
  count);
* search far looser than the tier (``-e 1000``) and apply the reporting ceiling
  (``E <= 10``) at reduce time, so the sensitivity curve comes out of one search;
* ``-s 7.5`` and ``--max-seqs 1000000``, and report the observed per-query
  maximum so prefilter censoring is visible rather than assumed away.

The reference is **eval2-v1** (577 = #225's 554 + #226's 23), not #225's ``v1``.
Because a complex is dropped when *either* subunit is contaminated, the unit
here is the accession, and ``selection.py`` applies it to both subunits.

    uv run python decontam_droplist.py --sequences '/data/.../sequences/*.fasta.gz' \\
        --reference data/eval2_reference.fasta --work /data/exp294_decontam
"""

import argparse
import csv
import glob
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
# Import #225's rule rather than restating it: two copies of a decontamination
# threshold drift, and the published corpora are defined by that module's
# numbers. It is stdlib-only, so it loads cleanly in this experiment's venv.
sys.path.insert(0, str(HERE.parent / "exp225_data_decontaminate_training_corpora"))

from decontam_lib import (  # pyrefly: ignore[missing-import]
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    ensure_mmseqs,
    is_sequence_contaminant,
)

from build_eval_reference import REFERENCE_VERSION

#: query, target, fractional identity, query coverage, e-value, bit score.
FORMAT = "query,target,fident,qcov,evalue,bits"
DROPLIST_SCHEMA = pa.schema(
    [("accession", pa.string()), ("eval_decontam_reason", pa.string())]
)


def run(cmd: list[str]) -> None:
    """Run a subprocess, echoing it. Searches here run for minutes to hours."""
    print("  $", " ".join(str(c) for c in cmd[:8]), "...", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def accession_from_header(header: str) -> str:
    """``AFDB:AF-<acc>-F1`` -> ``<acc>``; mirrors fetch_sequences."""
    token = header.split()[0]
    token = token.removeprefix("AFDB:")
    if token.startswith("AF-") and "-F" in token:
        return token[3 : token.rindex("-F")]
    fields = token.split("|")
    if len(fields) >= 3 and fields[0] in {"sp", "tr"}:
        return fields[1]
    return token


def search(
    target_db: Path,
    queries: Path,
    work: Path,
    *,
    sensitivity: float,
    max_seqs: int,
    search_evalue: float,
    threads: int,
    split_memory_limit: str,
) -> Path:
    """Search the reference against the AFCDB subunit DB, keeping every hit."""
    mmseqs = ensure_mmseqs()
    query_db = work / "queryDB"
    aln_db = work / "alnDB"
    tmp = work / "mmseqs_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for stale in list(work.glob("alnDB*")) + list(work.glob("queryDB*")):
        stale.unlink()

    run([mmseqs, "createdb", str(queries), str(query_db)])
    began = time.time()
    run(
        [
            mmseqs, "search", str(query_db), str(target_db), str(aln_db), str(tmp),
            "-s", str(sensitivity),
            "--max-seqs", str(max_seqs),
            "-e", str(search_evalue),
            "--threads", str(threads),
            "--split-memory-limit", split_memory_limit,
        ]
    )
    print(f"[mmseqs] search in {time.time() - began:.0f}s", flush=True)

    m8 = work / "aln_all_hits.m8"
    run(
        [
            mmseqs, "convertalis", str(query_db), str(target_db), str(aln_db), str(m8),
            "--format-output", FORMAT, "--threads", str(threads),
        ]
    )
    return m8


def build_droplist(m8: Path, report_ceiling: float) -> tuple[dict[str, str], dict[str, Any]]:
    """Fold every alignment into one drop record per contaminated accession."""
    dropped: dict[str, str] = {}
    best: dict[str, float] = {}
    alignments = 0
    per_query_alignments: Counter[str] = Counter()

    with m8.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            query, target, fident, qcov, evalue = row[0], row[1], *row[2:5]
            alignments += 1
            per_query_alignments[query] += 1
            identity, coverage, e = float(fident), float(qcov), float(evalue)
            if e > report_ceiling:
                continue
            if not is_sequence_contaminant(identity, coverage, e):
                continue
            accession = accession_from_header(target)
            # Keep the strongest hit's description so the reason is the reason,
            # not merely "some alignment somewhere cleared the bar".
            if accession not in best or e < best[accession]:
                best[accession] = e
                dropped[accession] = (
                    f"eval_homolog:{query}:identity={identity:.3f}:"
                    f"qcov={coverage:.3f}:evalue={e:.3g}"
                )

    censored = [q for q, n in per_query_alignments.items() if n >= 1_000_000]
    stats: dict[str, Any] = {
        "alignments_reported": alignments,
        "queries_with_alignments": len(per_query_alignments),
        "max_alignments_per_query": max(per_query_alignments.values(), default=0),
        "queries_at_max_seqs": len(censored),
        "dropped_accessions": len(dropped),
    }
    return dropped, stats


def build(
    sequences_glob: str,
    reference: Path,
    work: Path,
    out_path: Path,
    *,
    sensitivity: float = 7.5,
    max_seqs: int = 1_000_000,
    search_evalue: float = 1000.0,
    report_evalue_ceiling: float = 10.0,
    threads: int = 64,
    split_memory_limit: str = "200G",
    skip_search: bool = False,
) -> dict[str, Any]:
    """Create the AFCDB subunit DB, search eval2 against it, write the drop list."""
    work.mkdir(parents=True, exist_ok=True)
    mmseqs = ensure_mmseqs()
    target_db = work / "targetDB"
    if not skip_search:
        shards = sorted(Path(match) for match in glob.glob(sequences_glob))
        if not shards:
            raise FileNotFoundError(f"no sequence shards matched {sequences_glob}")
        for stale in work.glob("targetDB*"):
            stale.unlink()
        # mmseqs createdb reads gzipped FASTA directly and concatenates inputs.
        run([mmseqs, "createdb", *[str(s) for s in shards], str(target_db)])

    m8 = (
        work / "aln_all_hits.m8"
        if skip_search
        else search(
            target_db,
            reference,
            work,
            sensitivity=sensitivity,
            max_seqs=max_seqs,
            search_evalue=search_evalue,
            threads=threads,
            split_memory_limit=split_memory_limit,
        )
    )
    if not m8.is_file():
        raise FileNotFoundError(m8)

    dropped, stats = build_droplist(m8, report_evalue_ceiling)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accessions = sorted(dropped)
    pq.write_table(
        pa.table(
            {
                "accession": pa.array(accessions, type=pa.string()),
                "eval_decontam_reason": pa.array(
                    [dropped[a] for a in accessions], type=pa.string()
                ),
            },
            schema=DROPLIST_SCHEMA,
        ),
        out_path,
        compression="zstd",
    )
    provenance = {
        "reference": str(reference.resolve()),
        "reference_version": REFERENCE_VERSION,
        "sequences_glob": sequences_glob,
        "rule": (
            f"identity >= {SEQ_MIN_IDENTITY:.0%} over >= {SEQ_MIN_QCOV:.0%} query "
            f"coverage, or E <= {SEQ_MAX_EVALUE:g}"
        ),
        "search": {
            "sensitivity": sensitivity,
            "max_seqs": max_seqs,
            "search_evalue": search_evalue,
            "report_evalue_ceiling": report_evalue_ceiling,
            "mmseqs": mmseqs,
        },
        "output": str(out_path.resolve()),
        **stats,
    }
    out_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    if stats["queries_at_max_seqs"]:
        raise RuntimeError(
            f"{stats['queries_at_max_seqs']} queries hit --max-seqs {max_seqs:,}; "
            "their alignments are a censored sample and the drop list would be "
            "incomplete. Raise --max-seqs and re-run"
        )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, help="Glob of AFCDB FASTA shards.")
    parser.add_argument(
        "--reference", type=Path, default=HERE / "data/eval2_reference.fasta"
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("-s", "--sensitivity", type=float, default=7.5)
    parser.add_argument("--max-seqs", type=int, default=1_000_000)
    parser.add_argument("--search-evalue", type=float, default=1000.0)
    parser.add_argument("--report-evalue-ceiling", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--split-memory-limit", default="200G")
    parser.add_argument("--skip-search", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or args.work / "eval_accession_droplist.parquet"
    provenance = build(
        args.sequences,
        args.reference,
        args.work,
        out,
        sensitivity=args.sensitivity,
        max_seqs=args.max_seqs,
        search_evalue=args.search_evalue,
        report_evalue_ceiling=args.report_evalue_ceiling,
        threads=args.threads,
        split_memory_limit=args.split_memory_limit,
        skip_search=args.skip_search,
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
