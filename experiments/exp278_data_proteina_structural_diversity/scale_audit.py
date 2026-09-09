"""Audit a bounded stratified sample; never interpret its cap as a global cap."""

import argparse
import bisect
import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analyze_screen import (
    FIELDS,
    audit_prefilter,
    ca_structure,
    components,
    effective_clusters,
    external_structure_screen,
    retention_report,
    searches,
    secondary_fractions,
    sequence_screen,
    summarize_clusters,
    write_csv,
)
from launch import storage_filesystem


def accumulation_report(work: Path, report: Path) -> None:
    """Compare sample-size curves within each requested arm after exclusion."""
    with (report / "retention.csv").open() as handle:
        eligible = [
            row
            for row in csv.DictReader(handle)
            if row["quality_pass"] == "True"
            and row["sequence_excluded"] == row["structure_excluded"] == "False"
        ]
    results = []
    for geometry in ("original", "refolded"):
        with (work / f"{geometry}-pairs.tsv").open() as handle:
            edges = [
                (row["query"].split(".pdb")[0], row["target"].split(".pdb")[0])
                for row in csv.DictReader(
                    handle, fieldnames=FIELDS.split(","), delimiter="\t"
                )
                if min(float(row["qtmscore"]), float(row["ttmscore"])) >= 0.8
                and min(float(row["qcov"]), float(row["tcov"])) >= 0.8
            ]
        for condition in sorted({row["condition"] for row in eligible}):
            stems = [row["stem"] for row in eligible if row["condition"] == condition]
            rng = random.Random(27818)
            for count in sorted(
                {min(n, len(stems)) for n in (25, 50, 100, 200, 400, len(stems))}
            ):
                for repeat in range(20):
                    sample = rng.sample(stems, count)
                    results.append(
                        {
                            "geometry": geometry,
                            "condition": condition,
                            "sample_size": count,
                            "repeat": repeat,
                            "fine_clusters": len(components(sample, edges)),
                            "effective_clusters": effective_clusters(sample, edges),
                        }
                    )
    write_csv(report / "diversity-accumulation.csv", results)


def weighted_retention_report(report: Path) -> None:
    """Weight the exclusion audit to its sampled committed-refold population."""
    with (report / "retention.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with (report / "sample-populations.csv").open() as handle:
        populations = list(csv.DictReader(handle))
    total = sum(int(row["available_refolded"]) for row in populations)
    estimated = 0.0
    for population in populations:
        subset = [
            row
            for row in rows
            if 60 + 40 * ((int(row["length"]) - 60) // 40)
            == int(population["length_bin_start"])
            and row["condition"] == population["condition"]
        ]
        if len(subset) != int(population["sampled"]):
            raise ValueError("Audit retention rows do not match sample populations")
        passed = sum(
            row["quality_pass"] == "True"
            and row["sequence_excluded"] == row["structure_excluded"] == "False"
            for row in subset
        )
        estimated += int(population["available_refolded"]) * passed / len(subset)
    (report / "weighted-retention.json").write_text(
        json.dumps(
            {
                "committed_refold_population": total,
                "estimated_quality_and_reference_pass": estimated,
                "estimated_quality_and_reference_fraction": estimated / total,
                "scope": "Stratified point estimate among committed refolds at audit snapshot time; global fine-cluster capping is excluded. This is not final corpus retention.",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    fs = storage_filesystem("cw-us-east-02a")
    root = plan["output"].removeprefix("s3://")
    files = fs.find(root + "/cases", detail=True)
    cases = {case["id"]: case for case in plan["cases"]}
    strata = defaultdict(list)
    for path in sorted(files):
        if "/folded/candidates/" not in path or not path.endswith(".parquet"):
            continue
        marker = (
            path.replace("/candidates/", "/completed-parts/").removesuffix(".parquet")
            + ".json"
        )
        if marker not in files:
            continue
        case = cases[path.split("/cases/", 1)[1].split("/", 1)[0]]
        strata[((case["length"] - 60) // 40, case["condition"])].append((path, case))
    chosen = defaultdict(list)
    populations = []
    rng = random.Random(2781818)
    for (length_bin, condition), batches in sorted(strata.items()):
        ends = np.cumsum([case["batch_size"] for _, case in batches]).tolist()
        count = min(args.per_stratum, ends[-1])
        for index in rng.sample(range(ends[-1]), count):
            batch = bisect.bisect_right(ends, index)
            path, case = batches[batch]
            chosen[path].append(index - (ends[batch - 1] if batch else 0))
        populations.append(
            {
                "length_bin_start": 60 + 40 * length_bin,
                "condition": condition,
                "available_refolded": ends[-1],
                "sampled": count,
            }
        )
    transfer_bytes = sum(files[path]["size"] for path in chosen)
    if transfer_bytes > 9_000_000_000:
        raise ValueError("Audit transfer exceeds the 9 GB bound; reduce sample size")
    records = []
    timings = []
    for path, indices in sorted(chosen.items()):
        rows = pq.read_table(io.BytesIO(fs.cat(path))).to_pylist()
        case = cases[path.split("/cases/", 1)[1].split("/", 1)[0]]
        if len(rows) != case["batch_size"]:
            raise ValueError("Saved candidate batch differs from frozen manifest")
        selected = [rows[index] for index in indices]
        stems = {row["stem"] for row in selected}
        timing_path = (
            path.replace("/candidates/", "/timings/").removesuffix(".parquet") + ".csv"
        )
        sampling_path = (
            rows[0]["source_prefix"].removeprefix("s3://")
            + "/"
            + rows[0]["source_batch"]
        )
        transfer_bytes += files[timing_path]["size"] + files[sampling_path]["size"]
        if transfer_bytes > 9_000_000_000:
            raise ValueError("Audit including sampled timing archives exceeds 9 GB")
        timings.extend(
            row
            for row in csv.DictReader(io.StringIO(fs.cat(timing_path).decode()))
            if row["stem"] in stems
        )
        with np.load(io.BytesIO(fs.cat(sampling_path)), allow_pickle=False) as archive:
            sampled_timings = json.loads(str(archive["timings_json"]))
        for row in selected:
            timing = sampled_timings[row["sample_in_batch"]]
            timings.append(
                {**timing, "sampling_stem": timing["stem"], "stem": row["stem"]}
            )
        records.extend(
            {
                **rows[index],
                "audit_condition": case["condition"],
                "audit_model": case["model"],
            }
            for index in indices
        )
    if not records:
        raise ValueError("No committed refolds available for audit")
    args.work.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    with (args.report / "timings.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in timings for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(timings)
    pq.write_table(
        pa.Table.from_pylist(records),
        args.work / "candidates.parquet",
        compression="zstd",
    )
    quality = []
    for geometry in ("original", "refolded"):
        (args.work / geometry).mkdir(exist_ok=True)
    for row in records:
        sequence = row["sequence"]
        structure = gemmi.read_pdb_string(row["pdb_content"])
        refolded = np.asarray(
            [list(residue["CA"][0].pos) for residue in structure[0][0]]
        )
        summary = {
            key: row[key]
            for key in (
                "stem",
                "source_prefix",
                "scrmsd",
                "plddt",
                "quality_pass",
                "ca_clashes",
                "ca_chain_breaks",
            )
        }
        summary.update(
            length=len(sequence),
            condition=row["audit_condition"],
            model=row["audit_model"],
        )
        for geometry, coords in (
            ("original", np.asarray(row["original_ca"])),
            ("refolded", refolded),
        ):
            summary.update(
                {
                    f"{geometry}_{key}": value
                    for key, value in secondary_fractions(sequence, coords).items()
                }
            )
            ca_structure(sequence, coords).write_pdb(
                str(args.work / geometry / (row["stem"] + ".pdb"))
            )
        quality.append(summary)
    write_csv(args.report / "quality.csv", quality)
    write_csv(args.report / "sample-populations.csv", populations)
    provenance = {
        "seed": 2781818,
        "candidates": len(records),
        "downloaded_bytes": transfer_bytes,
        "selection": "Uniform candidate sampling within 40-aa length bins and requested arms, among committed refolds at snapshot time",
        "interpretation": "Stratified interim sample, not the full corpus. Retention and cluster cap below apply only to this sample; do not report them as final global retention.",
        "sources": sorted(chosen),
    }
    (args.report / "audit-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    searches(args.work, args.threads)
    sequence_screen(args.work, args.report, args.threads)
    summarize_clusters(args.work, args.report)
    audit_prefilter(args.work, args.report, workers=min(args.threads, 16))
    external_structure_screen(
        args.work,
        args.report,
        Path("/data/exp278/eval-structures"),
        Path("/data/exp225_decontam/afdb_reps_db/db/targetDB"),
        args.threads,
    )
    retention_report(args.work, args.report)
    accumulation_report(args.work, args.report)
    weighted_retention_report(args.report)
    for path in args.report.iterdir():
        if path.is_file():
            fs.pipe_file(
                f"{root}/reports/{args.report.name}/{path.name}", path.read_bytes()
            )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
