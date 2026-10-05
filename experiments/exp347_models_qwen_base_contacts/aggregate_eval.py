"""Validate complete per-protein outputs and aggregate canonical metric rows."""

import argparse
import csv
import io
import json
import math
from pathlib import Path

import fsspec


def read_json(uri: str) -> dict:
    """Read one object-store JSON artifact."""
    with fsspec.open(uri) as handle:
        return json.load(handle)


def write_json(uri: str, value: dict) -> None:
    """Commit a small JSON artifact after its constituent artifacts exist."""
    with fsspec.open(uri, "w") as handle:
        json.dump(value, handle, indent=2)


def write_csv(uri: str, rows: list[dict]) -> None:
    """Write complete metric/timing tables using injected storage credentials."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
    )
    writer.writeheader()
    writer.writerows(rows)
    with fsspec.open(uri, "w") as handle:
        handle.write(buffer.getvalue())


def aggregate(out: str, targets: Path, checkpoint: str, document_format: str) -> dict:
    """Require every unit and every metric before publishing aggregate R-precision."""
    records = [json.loads(line) for line in targets.read_text().splitlines()]
    timings, metrics = [], []
    for record in records:
        unit = f"{record['dataset']}__{record['stem']}"
        marker = read_json(f"{out}/{unit}/complete.json")
        if (
            marker["checkpoint"] != checkpoint
            or marker["document_format"] != document_format
        ):
            raise ValueError(f"Wrong checkpoint or document format: {unit}")
        if marker["n_rollouts"] != 100 or marker["unfinished_rollouts"] != 0:
            raise ValueError(f"Incomplete rollouts: {unit}")
        rows = marker["metrics"]
        expected = {
            (r, c)
            for r in ("all", "short", "medium", "long")
            for c in ("L", "L/2", "L/5", "R", "AUC")
        }
        if len(rows) != 20 or {(m["range"], m["cut"]) for m in rows} != expected:
            raise ValueError(f"Missing metric rows: {unit}")
        timings.append(marker["timing"])
        metrics.extend(
            {"dataset": record["dataset"], "stem": record["stem"], **m} for m in rows
        )
    means, valid = {}, {}
    for distance in ("all", "short", "medium", "long"):
        values = [
            m["precision"]
            for m in metrics
            if m["range"] == distance
            and m["cut"] == "R"
            and math.isfinite(m["precision"])
        ]
        if not values:
            raise ValueError(f"No valid {distance} R-precision")
        means[distance] = sum(values) / len(values)
        valid[distance] = len(values)
    summary = {
        "complete": True,
        "checkpoint": checkpoint,
        "document_format": document_format,
        "n_units": len(records),
        "n_rollouts": 100,
        "unfinished_rollouts": 0,
        "r_precision": means,
        "valid_r_precision": valid,
        "output": out,
    }
    write_csv(out + "/metrics.csv", metrics)
    write_csv(out + "/timings.csv", timings)
    if records[0]["eval_set"] == "legacy-e8-reference":
        summary["reference_passed"] = all(
            abs(means[k] - v) <= 0.005
            for k, v in {"all": 0.4245291, "long": 0.3656152}.items()
        )
        write_json(out + "/summary.json", summary)
        if not summary["reference_passed"]:
            raise ValueError(f"E8 reference outside tolerance: {means}")
    else:
        write_json(out + "/summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(args.out, args.targets, args.checkpoint, args.format), indent=2
        )
    )


if __name__ == "__main__":
    main()
