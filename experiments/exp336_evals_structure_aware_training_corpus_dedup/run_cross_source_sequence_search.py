# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Search current #232 AFDB sequences against current #232 ESM-Atlas.

Both inputs are already representatives of source-specific sequence clusters
(AFDB50 and ESM 40% Linclust). The high-value unknown is therefore cross-source
overlap. One permissive search at 30% identity and 80% coverage of *both* chains
supports every preregistered identity threshold by reduction.

The result DB retains alignment backtraces for later contact-map remapping, but
the first TSV intentionally omits alignment strings to keep the threshold audit
compact.
"""

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

FORMAT = "query,target,fident,qcov,tcov,evalue,bits,alnlen,qlen,tlen"
AFDB_DOCUMENTS = 3_963_003


def run(command: list[str]) -> float:
    """Run one visible command and return its wall time."""
    print("$", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    print(f"[done] {elapsed:.1f}s", flush=True)
    return elapsed


def database_ready(prefix: Path) -> bool:
    """Whether an MMseqs sequence DB prefix is complete enough to reuse."""
    return prefix.exists() and prefix.with_suffix(".dbtype").exists()


def build_database(mmseqs: str, fasta: Path, prefix: Path) -> float:
    """Create a stable, unshuffled MMseqs sequence database once."""
    if database_ready(prefix):
        print(f"[reuse] {prefix}", flush=True)
        return 0.0
    return run([mmseqs, "createdb", str(fasta), str(prefix), "--shuffle", "0"])


def sample_query_database(
    mmseqs: str,
    source_db: Path,
    work: Path,
    *,
    query_limit: int | None,
    seed: int,
) -> tuple[Path, float, str]:
    """Return the full query DB or a deterministic random createsubdb sample."""
    if query_limit is None:
        return source_db, 0.0, "full"
    if not 0 < query_limit <= AFDB_DOCUMENTS:
        raise ValueError(f"query_limit must be in [1, {AFDB_DOCUMENTS}]")
    tag = f"pilot{query_limit}-seed{seed}"
    subset = work / f"current_afdb_db_{tag}"
    if database_ready(subset):
        print(f"[reuse] {subset}", flush=True)
        return subset, 0.0, tag
    ids = work / f"current_afdb_db_{tag}.ids"
    selected = sorted(random.Random(seed).sample(range(AFDB_DOCUMENTS), query_limit))
    ids.write_text("".join(f"{key}\n" for key in selected))
    elapsed = run([mmseqs, "createsubdb", str(ids), str(source_db), str(subset)])
    return subset, elapsed, tag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmseqs",
        default="/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs",
    )
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=1000,
        help="prefilter cap per query; audit saturation before increasing",
    )
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--search-evalue", type=float, default=1000.0)
    parser.add_argument(
        "--query-limit",
        type=int,
        help="deterministic random AFDB pilot size; omit for the complete search",
    )
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    work = args.work
    query_fasta = work / "current_afdb.fasta"
    target_fasta = work / "current_esm_atlas.fasta"
    query_db = work / "current_afdb_db"
    target_db = work / "current_esm_atlas_db"
    for path in (query_fasta, target_fasta):
        if not path.exists():
            raise SystemExit(f"missing {path}; run build_current_fastas.py first")
    if not Path(args.mmseqs).exists():
        raise SystemExit(f"MMseqs2 binary does not exist: {args.mmseqs}")
    work.mkdir(parents=True, exist_ok=True)

    timings = {
        "createdb_afdb_seconds": build_database(args.mmseqs, query_fasta, query_db),
        "createdb_esm_seconds": build_database(args.mmseqs, target_fasta, target_db),
    }
    active_query_db, timings["createsubdb_seconds"], tag = sample_query_database(
        args.mmseqs,
        query_db,
        work,
        query_limit=args.query_limit,
        seed=args.seed,
    )
    suffix = "" if tag == "full" else f"_{tag}"
    result_db = work / f"cross_source_30id_80cov{suffix}_db"
    temporary = work / f"mmseqs_tmp_cross_source{suffix}"
    output = work / f"cross_source_30id_80cov{suffix}.tsv"
    if not result_db.with_suffix(".dbtype").exists():
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        timings["search_seconds"] = run(
            [
                args.mmseqs,
                "search",
                str(active_query_db),
                str(target_db),
                str(result_db),
                str(temporary),
                "-s",
                str(args.sensitivity),
                "--max-seqs",
                str(args.max_seqs),
                "-e",
                str(args.search_evalue),
                "--min-seq-id",
                "0.30",
                "-c",
                "0.80",
                "--cov-mode",
                "0",
                "-a",
                "1",
                "--threads",
                str(args.threads),
            ]
        )
    else:
        print(f"[reuse] {result_db}", flush=True)
        timings["search_seconds"] = 0.0
    if not output.exists():
        timings["convertalis_seconds"] = run(
            [
                args.mmseqs,
                "convertalis",
                str(active_query_db),
                str(target_db),
                str(result_db),
                str(output),
                "--format-output",
                FORMAT,
                "--threads",
                str(args.threads),
            ]
        )
    else:
        print(f"[reuse] {output}", flush=True)
        timings["convertalis_seconds"] = 0.0

    version = subprocess.run(
        [args.mmseqs, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    record = {
        "status": "complete",
        "mmseqs_version": version,
        "query_fasta": str(query_fasta),
        "query_database": str(active_query_db),
        "query_limit": args.query_limit,
        "query_sample_seed": args.seed if args.query_limit is not None else None,
        "target_fasta": str(target_fasta),
        "query_fasta_bytes": query_fasta.stat().st_size,
        "target_fasta_bytes": target_fasta.stat().st_size,
        "search": {
            "sensitivity": args.sensitivity,
            "max_seqs": args.max_seqs,
            "evalue": args.search_evalue,
            "min_sequence_identity": 0.30,
            "min_coverage": 0.80,
            "coverage_mode": 0,
            "coverage_semantics": "query and target",
            "backtrace": True,
        },
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "timings": timings,
    }
    (work / f"cross_source_sequence_search{suffix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
