"""Advance completed generation cases into bounded refolding workers."""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

from launch import storage_filesystem

HERE = Path(__file__).resolve().parent
ROOT = "s3://marin-us-east-02a/MarinFold/exp278-proteina"
CONDITIONS = ["unconditional", "1.x.x.x", "2.x.x.x", "3.x.x.x"]


def submit_fold(name: str, inputs: list[str], output: str) -> None:
    """Submit one independent refolding job with a three-hour hard timeout."""
    command = [
        "uv",
        "run",
        "python",
        "launch.py",
        "--name",
        name,
        "--checkpoints",
        "",
        "--no-wait",
        "--timeout",
        "10800",
        "--",
        "fold_worker.py",
        "--esm-revision",
        "75a3841ee059df2bf4d56688166c8fb459ddd97a",
    ]
    for prefix in inputs:
        command.extend(["--input", prefix])
    command.extend(["--output", output])
    subprocess.run(command, cwd=HERE, check=True)


def consolidate(fs, parent: str, children: list[str]) -> None:
    """Publish a length manifest only after every independently folded arm succeeds."""
    if fs.exists(parent + "/complete.json"):
        return
    if not all(fs.exists(child + "/complete.json") for child in children):
        return
    timings, results = [], []
    for child in children:
        results.append(json.loads(fs.cat(child + "/complete.json")))
        for kind in ["candidates", "documents-provisional"]:
            for source in fs.glob(child + f"/{kind}/*.parquet"):
                target = parent + f"/{kind}/" + Path(source).name
                fs.cp_file(source, target)
        with fs.open(child + "/timings.csv", "rt") as handle:
            timings.extend(csv.DictReader(handle))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(timings[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(timings)
    fs.pipe_file(parent + "/timings.csv", buffer.getvalue().encode())
    fs.pipe_file(
        parent + "/complete.json",
        json.dumps(
            {
                "candidates": sum(row["candidates"] for row in results),
                "quality_pass": sum(row["quality_pass"] for row in results),
                "children": children,
            }
        ).encode(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-folds", action="store_true")
    args = parser.parse_args()
    fs = storage_filesystem("cw-rno2a")
    status = []
    for length in [60, 100, 200, 300, 400, 500]:
        model = "short" if length < 300 else "long"
        prefixes = [
            f"{ROOT}/screen-{model}-v1/{model}-l{length}-{condition}"
            for condition in CONDITIONS
        ]
        ready = sum(
            fs.exists(prefix.removeprefix("s3://") + "/complete.json")
            for prefix in prefixes
        )
        name = f"exp278-fold-l{length}-v1"
        marker = f"{ROOT}/fold-l{length}-v1/complete.json".removeprefix("s3://")
        if length >= 400:
            children = []
            submitted = 0
            for index, prefix in enumerate(prefixes):
                part_name = f"exp278-fold-l{length}-part{index}-v1"
                output = f"{ROOT}/fold-l{length}-v1/part-{index}"
                children.append(output.removeprefix("s3://"))
                exists = (HERE / "data" / f"{part_name}-submission.json").exists()
                if (
                    args.launch_folds
                    and not exists
                    and fs.exists(prefix.removeprefix("s3://") + "/complete.json")
                ):
                    submit_fold(part_name, [prefix], output)
                    exists = True
                submitted += exists
            if args.launch_folds:
                consolidate(fs, marker.removesuffix("/complete.json"), children)
            row = {
                "length": length,
                "generated_cases": ready,
                "fold_parts_submitted": submitted,
                "fold_complete": fs.exists(marker),
            }
            if row["fold_complete"]:
                row.update(json.loads(fs.cat(marker)))
            status.append(row)
            continue
        row = {
            "length": length,
            "generated_cases": ready,
            "fold_submitted": (HERE / "data" / f"{name}-submission.json").exists(),
            "fold_complete": fs.exists(marker),
        }
        if row["fold_complete"]:
            row.update(json.loads(fs.cat(marker)))
            corrected_prefix = f"{ROOT}/fold-l{length}-cis-v2"
            corrected_marker = corrected_prefix.removeprefix("s3://") + "/complete.json"
            if fs.exists(corrected_marker):
                corrected = json.loads(fs.cat(corrected_marker))
                row["original_quality_pass"] = row["quality_pass"]
                row.update(
                    quality_pass=corrected["quality_pass"],
                    geometry_version=corrected["geometry_version"],
                    quality_prefix=corrected_prefix,
                )
        if args.launch_folds and ready == 4 and not row["fold_submitted"]:
            submit_fold(name, prefixes, f"{ROOT}/fold-l{length}-v1")
            row["fold_submitted"] = True
        status.append(row)
    print(json.dumps(status, indent=2))
    (HERE / "data/screen-status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
