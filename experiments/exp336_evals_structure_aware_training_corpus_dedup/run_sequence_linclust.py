# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Linear-time sequence-star candidate generation over the current #232 corpus.

The two source corpora are already source-specific sequence representatives.
Concatenating them and running Linclust finds the cross-source stars without a
3.96M-by-65.6M sensitive all-pairs search. ESM is concatenated first so the DB
layout is stable and source priority can be inspected, but downstream structure
selection does not trust transitive cluster membership: every proposed removal
will be explicitly realigned to its retained witness.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(command: list[str]) -> float:
    """Run a visible MMseqs command and return wall time."""
    print("$", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    print(f"[done] {elapsed:.1f}s", flush=True)
    return elapsed


def db_ready(prefix: Path) -> bool:
    """Whether a single-file or split MMseqs database is complete."""
    data_exists = prefix.exists() or Path(f"{prefix}.0").exists()
    return data_exists and prefix.with_suffix(".dbtype").exists()


def concat_database(mmseqs: str, first: Path, second: Path, output: Path) -> float:
    """Concatenate sequence or header DBs, renumbering the second key range."""
    if db_ready(output):
        print(f"[reuse] {output}", flush=True)
        return 0.0
    return run([mmseqs, "concatdbs", str(first), str(second), str(output)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmseqs",
        default="/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs",
    )
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument(
        "--evalue",
        type=float,
        default=1000.0,
        help="permissive reporting ceiling; identity and coverage define redundancy",
    )
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    if not 0.0 <= args.identity <= 1.0 or not 0.0 <= args.coverage <= 1.0:
        raise ValueError("identity and coverage must be in [0, 1]")

    work = args.work
    esm = work / "current_esm_atlas_db"
    afdb = work / "current_afdb_db"
    combined = work / "current_esm_then_afdb_db"
    for database in (esm, afdb):
        if not db_ready(database):
            raise SystemExit(f"missing {database}; run run_cross_source_sequence_search.py first")
    timings = {
        "concat_sequences_seconds": concat_database(args.mmseqs, esm, afdb, combined),
        "concat_headers_seconds": concat_database(
            args.mmseqs,
            Path(str(esm) + "_h"),
            Path(str(afdb) + "_h"),
            Path(str(combined) + "_h"),
        ),
    }
    evalue_tag = str(args.evalue).replace(".", "p")
    tag = (
        f"id{int(round(args.identity * 100)):03d}_"
        f"cov{int(round(args.coverage * 100)):03d}_e{evalue_tag}"
    )
    cluster_db = work / f"current_linclust_{tag}_db"
    temporary = work / f"linclust_tmp_{tag}"
    output = work / f"current_linclust_{tag}.tsv"
    if not db_ready(cluster_db):
        timings["linclust_seconds"] = run(
            [
                args.mmseqs,
                "linclust",
                str(combined),
                str(cluster_db),
                str(temporary),
                "--min-seq-id",
                str(args.identity),
                "-c",
                str(args.coverage),
                "--cov-mode",
                "0",
                "--cluster-mode",
                "0",
                "-e",
                str(args.evalue),
                "--threads",
                str(args.threads),
                "--remove-tmp-files",
                "1",
            ]
        )
    else:
        print(f"[reuse] {cluster_db}", flush=True)
        timings["linclust_seconds"] = 0.0
    if not output.exists():
        timings["createtsv_seconds"] = run(
            [
                args.mmseqs,
                "createtsv",
                str(combined),
                str(combined),
                str(cluster_db),
                str(output),
                "--threads",
                str(args.threads),
            ]
        )
    else:
        print(f"[reuse] {output}", flush=True)
        timings["createtsv_seconds"] = 0.0
    version = subprocess.run(
        [args.mmseqs, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    record = {
        "status": "complete_candidate_generation",
        "mmseqs_version": version,
        "database_order": ["esm_atlas", "afdb"],
        "documents": 69_516_181,
        "min_sequence_identity": args.identity,
        "min_bidirectional_coverage": args.coverage,
        "evalue_ceiling": args.evalue,
        "coverage_mode": 0,
        "cluster_mode": 0,
        "linclust_version": 2,
        "cluster_tsv": str(output),
        "cluster_tsv_bytes": output.stat().st_size,
        "timings": timings,
        "warning": (
            "Linclust membership is candidate generation. A final removal still requires a "
            "direct sequence realignment and the active structure/contact witness rule."
        ),
    }
    (work / f"current_linclust_{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
