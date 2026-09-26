# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Directly align every within-Linclust-star candidate pair.

Linclust is used only to organize candidate neighborhoods. Its clustering can
place members together through intermediate links, so cluster co-membership is
not direct sequence evidence. This script enumerates every unordered pair
within each star, rescores it with a local alignment, and keeps only pairs
satisfying the requested identity and bidirectional coverage. Enumerating all
intra-star pairs also lets a retained non-central structural mode serve as a
direct witness for later members of that mode.
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


def build_numeric_candidate_tsv(source: Path, output: Path) -> tuple[int, int, int]:
    """Enumerate every unordered intra-star pair in MMseqs prefilter TSV format."""
    candidate_pairs = 0
    membership_rows = 0
    clusters = 0
    temporary = output.with_suffix(output.suffix + ".partial")
    with source.open() as input_file, temporary.open("w") as output_file:
        active_representative: str | None = None
        members: list[str] = []

        def write_pairs() -> int:
            written = 0
            for left_index, left in enumerate(members):
                for right in members[left_index + 1 :]:
                    output_file.write(f"{left}\t{right}\t0\t0\n")
                    written += 1
            return written

        for line_number, line in enumerate(input_file, start=1):
            try:
                representative, member = line.rstrip("\n").split("\t")
            except ValueError as error:
                raise ValueError(f"malformed cluster row {line_number:,}: {line!r}") from error
            membership_rows += 1
            if active_representative is not None and representative != active_representative:
                candidate_pairs += write_pairs()
                clusters += 1
                members = []
            active_representative = representative
            members.append(member)
        if active_representative is not None:
            candidate_pairs += write_pairs()
            clusters += 1
    temporary.replace(output)
    return candidate_pairs, membership_rows, clusters


def parameter_tag(identity: float, coverage: float, evalue: float) -> str:
    """Return the filename tag shared with run_sequence_linclust.py."""
    evalue_tag = str(evalue).replace(".", "p")
    return (
        f"id{int(round(identity * 100)):03d}_"
        f"cov{int(round(coverage * 100)):03d}_e{evalue_tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmseqs",
        default="/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs",
    )
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--evalue", type=float, default=1000.0)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    if not 0.0 <= args.identity <= 1.0 or not 0.0 <= args.coverage <= 1.0:
        raise ValueError("identity and coverage must be in [0, 1]")

    tag = parameter_tag(args.identity, args.coverage, args.evalue)
    combined = args.work / "current_esm_then_afdb_db"
    clusters = args.work / f"current_linclust_{tag}_db"
    numeric_clusters = args.work / f"current_linclust_{tag}_numeric.tsv"
    numeric_pairs = args.work / f"current_linclust_{tag}_numeric_nonself.tsv"
    numeric_pairs_metadata = args.work / f"current_linclust_{tag}_numeric_nonself.json"
    prefilter = args.work / f"current_linclust_{tag}_direct_prefilter_db"
    direct = args.work / f"current_linclust_{tag}_direct_align_db"
    output = args.work / f"current_linclust_{tag}_direct_align.tsv"
    for database in (combined, clusters):
        if not db_ready(database):
            raise SystemExit(f"missing {database}; run run_sequence_linclust.py first")

    timings: dict[str, float] = {}
    if not numeric_clusters.exists():
        timings["export_numeric_cluster_seconds"] = run(
            [
                args.mmseqs,
                "prefixid",
                str(clusters),
                str(numeric_clusters),
                "--tsv",
                "1",
                "--threads",
                str(args.threads),
            ]
        )
    else:
        print(f"[reuse] {numeric_clusters}", flush=True)
        timings["export_numeric_cluster_seconds"] = 0.0

    started = time.perf_counter()
    if not numeric_pairs.exists():
        candidate_pairs, membership_rows, sequence_stars = build_numeric_candidate_tsv(
            numeric_clusters, numeric_pairs
        )
        numeric_pairs_metadata.write_text(
            json.dumps(
                {
                    "candidate_pairs": candidate_pairs,
                    "membership_rows": membership_rows,
                    "sequence_stars": sequence_stars,
                },
                indent=2,
            )
            + "\n"
        )
        timings["prepare_prefilter_tsv_seconds"] = time.perf_counter() - started
    else:
        print(f"[reuse] {numeric_pairs}", flush=True)
        pair_counts = json.loads(numeric_pairs_metadata.read_text())
        candidate_pairs = pair_counts["candidate_pairs"]
        membership_rows = pair_counts["membership_rows"]
        sequence_stars = pair_counts["sequence_stars"]
        timings["prepare_prefilter_tsv_seconds"] = 0.0

    if not db_ready(prefilter):
        timings["build_prefilter_db_seconds"] = run(
            [
                args.mmseqs,
                "tsv2db",
                str(numeric_pairs),
                str(prefilter),
                "--output-dbtype",
                "7",
            ]
        )
    else:
        print(f"[reuse] {prefilter}", flush=True)
        timings["build_prefilter_db_seconds"] = 0.0

    if not db_ready(direct):
        timings["direct_alignment_seconds"] = run(
            [
                args.mmseqs,
                "align",
                str(combined),
                str(combined),
                str(prefilter),
                str(direct),
                "--alignment-mode",
                "3",
                "--min-seq-id",
                str(args.identity),
                "-c",
                str(args.coverage),
                "--cov-mode",
                "0",
                "-e",
                str(args.evalue),
                "-a",
                "1",
                "--threads",
                str(args.threads),
            ]
        )
    else:
        print(f"[reuse] {direct}", flush=True)
        timings["direct_alignment_seconds"] = 0.0

    if not output.exists():
        timings["convertalis_seconds"] = run(
            [
                args.mmseqs,
                "convertalis",
                str(combined),
                str(combined),
                str(direct),
                str(output),
                "--format-output",
                "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov",
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
        "status": "complete_direct_sequence_verification",
        "mmseqs_version": version,
        "candidate_cluster_db": str(clusters),
        "candidate_pairs": candidate_pairs,
        "membership_rows": membership_rows,
        "sequence_stars": sequence_stars,
        "direct_alignment_tsv": str(output),
        "direct_alignment_tsv_bytes": output.stat().st_size,
        "min_sequence_identity": args.identity,
        "min_bidirectional_coverage": args.coverage,
        "evalue_ceiling": args.evalue,
        "coverage_mode": 0,
        "sequence_identity_mode": 0,
        "alignment": "MMseqs2 Smith-Waterman local alignment (alignment-mode 3)",
        "timings": timings,
        "warning": (
            "A directly verified sequence pair is only eligible for removal. "
            "The structure/contact witness rule must still be applied."
        ),
    }
    (args.work / f"current_linclust_{tag}_direct.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
