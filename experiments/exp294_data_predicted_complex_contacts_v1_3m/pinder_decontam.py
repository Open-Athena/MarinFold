# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage E — decontaminate the PINDER arm against the eval2-v1 reference.

The AFCDB arm decontaminated *before* extraction, on UniProt accessions, because
extraction from EBI was the expensive step and a drop list let us avoid paying
for documents we would discard. PINDER inverts both facts:

* fetching is cheap (a byte offset against a CDN), so generate-then-filter costs
  little; and
* **the UniProt sequence is not what was solved.** A PDB chain is a crystal
  construct -- tags, truncations, unresolved loops -- so an accession-level drop
  list would decontaminate against a sequence the structure does not contain.

So this runs on the sequences carried on the generated documents, which are the
residues actually present in the deposited structure. The rule, search depth and
reporting ceiling are #225's, imported from ``decontam_lib`` exactly as
``decontam_droplist.py`` does for AFCDB, so both arms are filtered to the same
standard.

A document is dropped if **either** chain is contaminated, matching the AFCDB
arm's per-subunit policy.

    uv run python pinder_decontam.py --documents '/data/.../pinder/corpus/documents/*.parquet' \\
        --reference data/eval2_reference.fasta --work /data/exp294_pinder_decontam
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

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp225_data_decontaminate_training_corpora"))

from decontam_lib import (  # pyrefly: ignore[missing-import]
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    ensure_mmseqs,
    is_sequence_contaminant,
)

from build_eval_reference import REFERENCE_VERSION

FORMAT = "query,target,fident,qcov,evalue,bits"
DROP_SCHEMA = pa.schema(
    [("system_id", pa.string()), ("eval_decontam_reason", pa.string())]
)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_cmd(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd[:8]), "...", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def write_chain_fasta(documents_glob: str, path: Path) -> tuple[int, int]:
    """One FASTA record per chain, named ``<system_id>#R`` / ``#L``.

    The chain suffix survives into the mmseqs target name, so a hit maps
    straight back to the document without a join.
    """
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT system_id, sequence_R, sequence_L
        FROM read_parquet({_sql_literal(documents_glob)})
        """
    ).arrow().read_all().to_pylist()
    systems = 0
    chains = 0
    with path.open("w") as handle:
        for row in rows:
            systems += 1
            for tag, seq in (("R", row["sequence_R"]), ("L", row["sequence_L"])):
                if not seq:
                    continue
                clean = "".join(c for c in seq.upper() if c.isalpha())
                if not clean:
                    continue
                handle.write(f">{row['system_id']}#{tag}\n{clean}\n")
                chains += 1
    return systems, chains


def build(
    documents_glob: str,
    reference: Path,
    work: Path,
    out_path: Path,
    *,
    sensitivity: float = 7.5,
    max_seqs: int = 1_000_000,
    search_evalue: float = 1000.0,
    report_evalue_ceiling: float = 10.0,
    threads: int = 56,
    split_memory_limit: str = "200G",
) -> dict[str, Any]:
    """Search eval2-v1 against the arm's own chain sequences; drop either-chain hits."""
    work.mkdir(parents=True, exist_ok=True)
    if not glob.glob(documents_glob):
        raise FileNotFoundError(f"no documents matched {documents_glob}")
    mmseqs = ensure_mmseqs()

    fasta = work / "pinder_chains.fasta"
    systems, chains = write_chain_fasta(documents_glob, fasta)
    print(f"[decontam] {systems:,} systems -> {chains:,} chain sequences", flush=True)

    target_db, query_db = work / "targetDB", work / "queryDB"
    aln_db, tmp = work / "alnDB", work / "mmseqs_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for stale in list(work.glob("targetDB*")) + list(work.glob("alnDB*")) + list(work.glob("queryDB*")):
        stale.unlink()
    run_cmd([mmseqs, "createdb", str(fasta), str(target_db)])
    run_cmd([mmseqs, "createdb", str(reference), str(query_db)])
    began = time.time()
    run_cmd([
        mmseqs, "search", str(query_db), str(target_db), str(aln_db), str(tmp),
        "-s", str(sensitivity), "--max-seqs", str(max_seqs), "-e", str(search_evalue),
        "--threads", str(threads), "--split-memory-limit", split_memory_limit,
    ])
    print(f"[mmseqs] search in {time.time()-began:.0f}s", flush=True)
    m8 = work / "aln_all_hits.m8"
    run_cmd([mmseqs, "convertalis", str(query_db), str(target_db), str(aln_db),
             str(m8), "--format-output", FORMAT, "--threads", str(threads)])

    dropped: dict[str, str] = {}
    best: dict[str, float] = {}
    alignments = 0
    per_query: Counter[str] = Counter()
    with m8.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            query, target = row[0], row[1]
            identity, coverage, evalue = float(row[2]), float(row[3]), float(row[4])
            alignments += 1
            per_query[query] += 1
            if evalue > report_evalue_ceiling:
                continue
            if not is_sequence_contaminant(identity, coverage, evalue):
                continue
            system_id = target.rsplit("#", 1)[0]
            if system_id not in best or evalue < best[system_id]:
                best[system_id] = evalue
                dropped[system_id] = (
                    f"eval_homolog:{query}:chain={target.rsplit('#',1)[-1]}:"
                    f"identity={identity:.3f}:qcov={coverage:.3f}:evalue={evalue:.3g}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(dropped)
    pq.write_table(
        pa.table({"system_id": pa.array(ids, type=pa.string()),
                  "eval_decontam_reason": pa.array([dropped[i] for i in ids], type=pa.string())},
                 schema=DROP_SCHEMA),
        out_path, compression="zstd",
    )
    censored = [q for q, n in per_query.items() if n >= max_seqs]
    stats = {
        "documents_glob": documents_glob,
        "reference": str(reference.resolve()),
        "reference_version": REFERENCE_VERSION,
        "rule": (f"identity >= {SEQ_MIN_IDENTITY:.0%} over >= {SEQ_MIN_QCOV:.0%} query "
                 f"coverage, or E <= {SEQ_MAX_EVALUE:g}"),
        "systems": systems,
        "chain_sequences": chains,
        "alignments_reported": alignments,
        "queries_with_alignments": len(per_query),
        "max_alignments_per_query": max(per_query.values(), default=0),
        "queries_at_max_seqs": len(censored),
        "dropped_systems": len(dropped),
        "dropped_fraction": round(len(dropped) / max(systems, 1), 5),
        "droplist": str(out_path.resolve()),
    }
    out_path.with_suffix(".provenance.json").write_text(json.dumps(stats, indent=2) + "\n")
    if censored:
        raise RuntimeError(
            f"{len(censored)} queries hit --max-seqs {max_seqs:,}; the drop list "
            "would be a censored sample. Raise --max-seqs and re-run"
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--reference", type=Path, default=HERE / "data/eval2_reference.fasta")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("-s", "--sensitivity", type=float, default=7.5)
    parser.add_argument("--max-seqs", type=int, default=1_000_000)
    parser.add_argument("--search-evalue", type=float, default=1000.0)
    parser.add_argument("--report-evalue-ceiling", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument("--split-memory-limit", default="200G")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or args.work / "pinder_eval_droplist.parquet"
    print(json.dumps(build(
        args.documents, args.reference, args.work, out,
        sensitivity=args.sensitivity, max_seqs=args.max_seqs,
        search_evalue=args.search_evalue, report_evalue_ceiling=args.report_evalue_ceiling,
        threads=args.threads, split_memory_limit=args.split_memory_limit,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
