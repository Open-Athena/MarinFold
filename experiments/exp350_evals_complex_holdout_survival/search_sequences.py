"""Run a pinned sensitive sequence audit and preserve full alignment evidence."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq

MMSEQS = Path.home() / ".cache/marinfold/mmseqs/mmseqs/bin/mmseqs"
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits,nident,qlen,tlen,qstart,qend,tstart,tend"


def run(command: list[str | Path], log_path: Path | None = None) -> None:
    """Run a command, preserving its diagnostic output and failing on errors."""
    print(" ".join(map(str, command)), flush=True)
    if log_path is None:
        subprocess.run(list(map(str, command)), check=True)
        return
    with log_path.open("w") as log:
        subprocess.run(
            list(map(str, command)), check=True, stdout=log, stderr=subprocess.STDOUT
        )


def clear_mmseqs_prefix(prefix: Path) -> None:
    """Remove this experiment's prior hash-tagged MMseqs DB or temporary tree."""
    for path in prefix.parent.glob(f"{prefix.name}*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def native_database(work: Path) -> Path:
    """Select exp225's retained rows from the complete pre-decontamination index."""
    source = Path("/data/exp213_overlap/targetDB")
    target = work / "nativeDB"
    if target.with_suffix(".dbtype").exists():
        return target
    drop_path = Path("/data/exp225_decontam/droplist_final.parquet")
    drops: dict[str, set[str]] = {}
    for batch in pq.ParquetFile(drop_path).iter_batches(columns=["arm", "entry_id"]):
        for row in batch.to_pylist():
            drops.setdefault(row["arm"], set()).add(row["entry_id"])
    keys = work / "native_keys.txt"
    counts: dict[str, int] = {}
    rejected: dict[str, int] = {}
    with source.with_suffix(".lookup").open() as fh, keys.open("w") as out:
        for line in fh:
            key, name, _ = line.rstrip("\n").split("\t")
            arm, name = name.split("|", 1)
            entry = name.split("_", 2)[2]
            if entry in drops[arm]:
                rejected[arm] = rejected.get(arm, 0) + 1
                continue
            counts[arm] = counts.get(arm, 0) + 1
            out.write(key + "\n")
    if counts != {"afdb": 3963003, "esm_atlas": 65553178}:
        raise ValueError(f"Unexpected retained native corpus counts: {counts}")
    run([MMSEQS, "createsubdb", keys, source, target, "--subdb-mode", "1"])
    (work / "native_db_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "droplist": str(drop_path),
                "counts": counts,
                "rejected": rejected,
                "droplist_sha256": hashlib.file_digest(
                    drop_path.open("rb"), "sha256"
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return target


def main() -> None:
    """Build the requested index and search candidates; do not infer passes from absent files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["native", "complex", "helico"], required=True)
    ap.add_argument("--queries", type=Path, default=Path("data/queries.fasta"))
    ap.add_argument("--work", type=Path, default=Path("/data/exp350"))
    ap.add_argument(
        "--target-db",
        type=Path,
        help="Reuse an existing MMseqs target database instead of building one in --work.",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        help="Override the arm-specific MMseqs prefilter result-list cap.",
    )
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.target_db is not None:
        target = args.target_db
        if not target.with_suffix(".dbtype").exists():
            raise FileNotFoundError(f"Missing MMseqs target database: {target}")
    elif args.arm == "native":
        target = native_database(args.work)
    elif args.arm == "complex":
        target = args.work / "complexDB"
        if not target.with_suffix(".dbtype").exists():
            sources = sorted((args.work / "complex_sequences").glob("*.fasta"))
            if len(sources) != 170:
                raise ValueError(f"Expected 170 FASTAs, got {len(sources)}")
            run([MMSEQS, "createdb", *sources, target, "--shuffle", "0"])
    else:
        target = args.work / "helicoDB"
        if not target.with_suffix(".dbtype").exists():
            run(
                [
                    MMSEQS,
                    "createdb",
                    args.work / "helico_training_pool.fasta",
                    target,
                    "--shuffle",
                    "0",
                ]
            )
    query_sha256 = hashlib.file_digest(args.queries.open("rb"), "sha256").hexdigest()
    run_tag = f"v3_{query_sha256[:12]}"
    query = args.work / f"{args.arm}_queryDB_{run_tag}"
    clear_mmseqs_prefix(query)
    run([MMSEQS, "createdb", args.queries, query, "--shuffle", "0"])
    start = time.monotonic()
    searches = []
    max_seqs = args.max_seqs or (500000 if args.arm == "helico" else 100000)
    for coverage_name, coverage_mode in [("query", "2"), ("target", "1")]:
        result = args.work / f"{args.arm}_{coverage_name}_alnDB_{run_tag}"
        output = args.work / f"{args.arm}_{coverage_name}_alignments.tsv"
        log_path = args.work / f"{args.arm}_{coverage_name}_search.log"
        temporary = args.work / f"{args.arm}_{coverage_name}_tmp_{run_tag}"
        clear_mmseqs_prefix(result)
        clear_mmseqs_prefix(temporary)
        output.unlink(missing_ok=True)
        command = [
            MMSEQS,
            "search",
            query,
            target,
            result,
            temporary,
            "-s",
            "7.5",
            "-e",
            "1000",
            "--max-seqs",
            str(max_seqs),
            "--min-seq-id",
            "0.30",
            "--alignment-mode",
            "3",
            "--seq-id-mode",
            "0",
            "-a",
            "1",
            "-c",
            "0.50",
            "--cov-mode",
            coverage_mode,
            "--threads",
            str(args.threads),
            "--split-memory-limit",
            "80G",
        ]
        run(command, log_path)
        log = log_path.read_text()
        overflow_counts = [int(n) for n in re.findall(r"(\d+) overflows", log)]
        median_result_counts = [
            int(n) for n in re.findall(r"(\d+) median result list length", log)
        ]
        run(
            [
                MMSEQS,
                "convertalis",
                query,
                target,
                result,
                output,
                "--format-output",
                FORMAT,
                "--threads",
                str(args.threads),
            ]
        )
        searches.append(
            {
                "coverage": coverage_name,
                "coverage_mode": int(coverage_mode),
                "command": list(map(str, command)),
                "log": str(log_path),
                "overflow_counts": overflow_counts,
                "median_result_list_lengths": median_result_counts,
                "max_seqs": max_seqs,
                "prefilter_median_at_cap": any(
                    count >= max_seqs for count in median_result_counts
                ),
                "output": str(output),
            }
        )
    (args.work / f"{args.arm}_search.json").write_text(
        json.dumps(
            {
                "searches": searches,
                "format": FORMAT,
                "elapsed_seconds": time.monotonic() - start,
                "version": subprocess.check_output(
                    [str(MMSEQS), "version"], text=True
                ).strip(),
                "queries_sha256": query_sha256,
                "queries": str(args.queries.resolve()),
                "target": str(target),
                "rule": "identity >=0.30 over >=0.50 coverage of shorter sequence; max(qcov,tcov)",
                "search_evalue_ceiling": 1000,
                "coverage_filtered_at_search_time": True,
                "coverage_union": "query coverage >=0.50 OR target coverage >=0.50",
                "exact_identity_counts": "nident stored by MMseqs with -a 1 backtraces",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
