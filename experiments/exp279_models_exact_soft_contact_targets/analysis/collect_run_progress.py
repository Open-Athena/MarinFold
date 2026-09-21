# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect exp279's own step-vs-wall-clock progress, for the reliability view."""

import csv
import datetime
import math
from pathlib import Path

import wandb

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = "open-athena/MarinFold"
RUN = "exp279-soft-s0-cw-h100x32-b02"


def main() -> None:
    api = wandb.Api(timeout=90)
    run = api.run(f"{PROJECT}/{RUN}")
    frame = run.history(
        keys=["global_step", "_timestamp", "throughput/duration"], samples=200000
    )
    frame = frame.dropna(subset=["global_step", "_timestamp"])
    # Column access by name: itertuples renames "_timestamp" and
    # "throughput/duration" positionally, which is fragile.
    steps = frame["global_step"].tolist()
    stamps = frame["_timestamp"].tolist()
    durations = (
        frame["throughput/duration"].tolist()
        if "throughput/duration" in frame.columns
        else [float("nan")] * len(steps)
    )
    rows = [
        {
            "global_step": int(step),
            "utc": datetime.datetime.fromtimestamp(
                float(stamp), datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            "seconds_per_update": (
                "" if duration is None or math.isnan(float(duration)) else f"{float(duration):.4f}"
            ),
        }
        for step, stamp, duration in zip(steps, stamps, durations, strict=True)
    ]
    rows.sort(key=lambda r: r["global_step"])
    output = EXPERIMENT / "data/run_progress.csv"
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} progress samples through step {rows[-1]['global_step']} to {output}")


if __name__ == "__main__":
    main()
