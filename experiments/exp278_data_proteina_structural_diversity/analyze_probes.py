"""Summarize numerical precision and matched-backbone sequence retry probes."""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analyze_screen import write_csv
from launch import storage_filesystem
from quality import aligned_rmsd

ROOT = "marin-us-east-02a/MarinFold/exp278-proteina"
HERE = Path(__file__).resolve().parent


def read_candidates(fs, suffix: str) -> list[dict]:
    root = ROOT + "/" + suffix
    if not fs.exists(root + "/complete.json"):
        raise ValueError(f"Incomplete probe: {suffix}")
    rows = []
    for path in sorted(fs.glob(root + "/candidates/*.parquet")):
        with fs.open(path, "rb") as handle:
            rows.extend(pq.read_table(handle).to_pylist())
    return rows


def main() -> None:
    fs = storage_filesystem("cw-rno2a")
    groups = defaultdict(list)
    for row in read_candidates(fs, "fold-precision-cis-v2"):
        groups[row["source_prefix"].rsplit("/", 1)[-1]].append(row)
    arms = []
    by_key = {}
    for name, rows in groups.items():
        timing_bytes = fs.cat(f"{ROOT}/{name}/timings.csv")
        timings = list(csv.DictReader(io.StringIO(timing_bytes.decode())))
        fs.get_file(
            f"{ROOT}/{name}/timings.csv", str(HERE / f"data/{name}-timings.csv")
        )
        arms.append(
            {
                "arm": name,
                "candidates": len(rows),
                "quality_pass": sum(row["quality_pass"] for row in rows),
                "median_scrmsd": float(np.median([row["scrmsd"] for row in rows])),
                "median_plddt": float(np.median([row["plddt"] for row in rows])),
                "warm_sampling_seconds": float(
                    np.mean(
                        [
                            float(row["elapsed_seconds"])
                            for row in timings
                            if int(row["batch_index"]) > 0
                        ]
                    )
                ),
            }
        )
        by_key[name] = {
            (row["source_batch"], row["sample_in_batch"]): row for row in rows
        }
    reference = by_key["long500-batch24-v1"]
    accelerated = by_key["long500-tf32-v1"]
    if set(reference) != set(accelerated):
        raise ValueError("Numerical probes have different backbone seed counts")
    pairs = [
        {
            "batch": key[0],
            "sample_in_batch": key[1],
            "reference_stem": row["stem"],
            "tf32_stem": accelerated[key]["stem"],
            "original_ca_rmsd": aligned_rmsd(
                np.asarray(row["original_ca"]),
                np.asarray(accelerated[key]["original_ca"]),
            ),
        }
        for key, row in reference.items()
    ]
    write_csv(HERE / "data/precision-screen/arms.csv", arms)
    write_csv(HERE / "data/precision-screen/coordinate-pairs.csv", pairs)

    first = {row["stem"]: row for row in read_candidates(fs, "fold-l500-v1/part-0")}
    second = read_candidates(fs, "fold-second-attempt-v1")
    retry_rows = [
        {
            "stem": row["stem"],
            "first_quality_pass": first[row["stem"]]["quality_pass"],
            "second_quality_pass": row["quality_pass"],
            "best_of_two_quality_pass": first[row["stem"]]["quality_pass"]
            or row["quality_pass"],
            "first_scrmsd": first[row["stem"]]["scrmsd"],
            "second_scrmsd": row["scrmsd"],
        }
        for row in second
    ]
    write_csv(HERE / "data/retry-screen/paired-quality.csv", retry_rows)
    summary = {
        "backbones": len(retry_rows),
        "first_quality_pass": sum(row["first_quality_pass"] for row in retry_rows),
        "second_quality_pass": sum(row["second_quality_pass"] for row in retry_rows),
        "best_of_two_quality_pass": sum(
            row["best_of_two_quality_pass"] for row in retry_rows
        ),
        "incremental_quality_pass": sum(
            not row["first_quality_pass"] and row["second_quality_pass"]
            for row in retry_rows
        ),
        "scope": "32 identical unconditional 500-aa backbones; quality only, before decontamination and diversity filtering",
    }
    for name in ["fold-precision-cis-v2", "fold-second-attempt-v1"]:
        fs.get_file(
            f"{ROOT}/{name}/timings.csv", str(HERE / f"data/{name}-timings.csv")
        )
    timing = list(
        csv.DictReader(
            io.StringIO(fs.cat(f"{ROOT}/fold-second-attempt-v1/timings.csv").decode())
        )
    )
    seconds = sum(float(row["elapsed_seconds"]) for row in timing)
    summary["additional_design_and_refold_gpu_seconds"] = seconds
    summary["gpu_seconds_per_incremental_quality_pass"] = (
        seconds / summary["incremental_quality_pass"]
        if summary["incremental_quality_pass"]
        else None
    )
    (HERE / "data/retry-screen/summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (HERE / "data/precision-screen/interpretation.json").write_text(
        json.dumps(
            {
                "raw_coordinates": "paired backbone seeds, reference versus TF32",
                "quality_comparison": "48 per arm; MPNN seeds differ between numerical arms, so refolding quality is not a paired test",
                "decision": "promising throughput result; small quality comparison does not establish noninferiority",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"precision": arms, "retry": summary}, indent=2))


if __name__ == "__main__":
    main()
