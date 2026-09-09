"""Recluster all screen lengths together, reusing identical reference searches."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from analyze_screen import (
    FIELDS,
    FOLDSEEK,
    audit_prefilter,
    collect,
    retention_report,
    run,
    summarize_clusters,
    write_csv,
)

HERE = Path(__file__).resolve().parent
LENGTHS = [60, 100, 200, 300, 400, 500]


def fine_searches(work: Path) -> None:
    """Reuse within-length hits and search the only feasible cross-length pair.

    A TM score normalized by the longer chain cannot exceed Lshort/Llong:
    each aligned residue contributes at most one. On this six-length grid,
    only 400 versus 500 can reach 0.8. This bound applies to fine redundancy,
    not the exploratory broad metric, which remains a per-length result.
    """
    for geometry in ["original", "refolded"]:
        cross = work / f"{geometry}-cross-400-500.tsv"
        run(
            [
                FOLDSEEK,
                "easy-search",
                Path("/data/exp278/l400-screen") / geometry,
                Path("/data/exp278/l500-screen") / geometry,
                cross,
                work / f"{geometry}-cross-tmp",
                "--alignment-type",
                "1",
                "--format-output",
                FIELDS,
                "--max-seqs",
                "10000",
                "-e",
                "100",
                "-s",
                "9.5",
                "--threads",
                "24",
            ],
            work / f"{geometry}-cross-search.log",
        )
        with (work / f"{geometry}-pairs.tsv").open("wb") as output:
            for path in [
                *(
                    Path(f"/data/exp278/l{length}-screen/{geometry}-pairs.tsv")
                    for length in LENGTHS
                ),
                cross,
            ]:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["all", "prepare", "finalize"], default="all"
    )
    args = parser.parse_args()
    work = Path("/data/exp278/full-screen")
    report = HERE / "data/full-screen"
    root = "s3://marin-us-east-02a/MarinFold/exp278-proteina"
    prefixes = [
        f"{root}/fold-l{length}-{'cis-v2' if length <= 300 else 'v1'}"
        for length in LENGTHS
    ]
    if args.phase != "finalize":
        collect(prefixes, work, report)
        fine_searches(work)
        summarize_clusters(work, report, thresholds=(("fine", 0.8),))
        audit_prefilter(work, report, workers=16, thresholds=(0.8,))
    if args.phase == "prepare":
        return
    # Query direction and reference databases are identical across length jobs;
    # concatenating their hits preserves the original E-values and search caps.
    for length in LENGTHS:
        source = HERE / f"data/l{length}-screen"
        if not (source / "retention.json").exists():
            raise ValueError(f"Incomplete length report: {length}")
    for name in [
        "sequence-exclusions.csv",
        "structure-exclusions.csv",
        "nearest-eval-structure.csv",
        "nearest-training-structure.csv",
    ]:
        if name == "nearest-training-structure.csv" and any(
            not (HERE / f"data/l{length}-screen" / name).exists() for length in LENGTHS
        ):
            continue
        rows = []
        for length in LENGTHS:
            path = HERE / f"data/l{length}-screen" / name
            if not path.exists() and name.startswith("nearest-"):
                raise ValueError(f"Incomplete reference report: {path}")
            if path.exists():
                with path.open() as handle:
                    rows.extend(csv.DictReader(handle))
        write_csv(report / name, rows)
    for name in ["sequence-screen.json", "structure-screen.json"]:
        sources = [
            json.loads((HERE / f"data/l{length}-screen" / name).read_text())
            for length in LENGTHS
        ]
        merged = {
            **sources[0],
            "candidates": 1536,
            "excluded_candidates": sum(row["excluded_candidates"] for row in sources),
            "source_reports": [f"l{length}-screen/{name}" for length in LENGTHS],
        }
        (report / name).write_text(json.dumps(merged, indent=2) + "\n")
    retention_report(work, report)
    training_path = report / "nearest-training-structure.csv"
    if training_path.exists():
        with (report / "retention.csv").open() as handle:
            selected = {
                row["stem"]: row
                for row in csv.DictReader(handle)
                if row["selected"] == "True"
            }
        with training_path.open() as handle:
            hits = {row["stem"]: row for row in csv.DictReader(handle)}
        summaries = []
        for length in ["all", *map(str, LENGTHS)]:
            stems = [
                stem
                for stem, row in selected.items()
                if length == "all" or row["length"] == length
            ]
            found = [hits[stem] for stem in stems if stem in hits]
            scores = [float(row["min_tm"]) for row in found]
            summaries.append(
                {
                    "length": length,
                    "selected": len(stems),
                    "with_detected_match": len(found),
                    "min_tm_p10": float(np.quantile(scores, 0.1)),
                    "min_tm_median": float(np.median(scores)),
                    "min_tm_p90": float(np.quantile(scores, 0.9)),
                    "detected_fine_neighbor": sum(
                        float(row["min_tm"]) >= 0.8
                        and min(float(row["qcov"]), float(row["tcov"])) >= 0.8
                        for row in found
                    ),
                }
            )
        write_csv(report / "training-comparison-summary.csv", summaries)
    (report / "analysis-provenance.json").write_text(
        json.dumps(
            {
                "scope": "all 1536 initial-screen candidates; cross-length clustering",
                "cross_length_bound": "min TM <= shorter length / longer length; only 400 versus 500 can meet fine threshold 0.8; broad metrics are per-length only",
                "reference_searches": "reused per-length searches with identical fixed targets",
                "prefilter_audit": "64 hash-selected candidates, exhaustive TM-align",
                "training_reference_comparison_complete": all(
                    (HERE / f"data/l{length}-screen/training-reference.json").exists()
                    for length in LENGTHS
                ),
                "remaining_pilot_arms": "A/T labels, higher noise, 400M triangle control, checkpoint crossover, and larger saturation study were not run",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
