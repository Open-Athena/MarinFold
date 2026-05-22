# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Append (or upsert by ``experiment_id``) one row in ``data/experiments.tsv``.

Used by every algorithm we try so the experiment ledger stays
consistent. The runtime ratio + LDDT delta are computed automatically
relative to the row keyed ``baseline_naive`` (which is the row
``run_baseline.py`` writes).

Library surface (preferred — keeps logic in-process):

    from append_experiment_row import upsert_experiment_row
    upsert_experiment_row(
        experiment_id="foo_v1",
        description="...",
        mean_lddt=0.31,
        median_lddt=0.28,
        total_wall_seconds=1234.5,
        notes="",
    )

CLI (one-off / shell):

    uv run python append_experiment_row.py \
        --experiment-id foo_v1 \
        --description "..." \
        --mean-lddt 0.31 --median-lddt 0.28 \
        --total-wall-seconds 1234.5
"""

import argparse
import csv
from pathlib import Path


_THIS = Path(__file__).resolve().parent
_DEFAULT_TSV = _THIS / "data" / "experiments.tsv"

FIELDNAMES = [
    "experiment_id", "description",
    "mean_lddt", "median_lddt",
    "total_wall_seconds", "runtime_ratio_vs_baseline",
    "mean_lddt_delta_pct", "notes",
]
BASELINE_ID = "baseline_naive"


def _load(tsv_path: Path) -> list[dict]:
    if not tsv_path.exists():
        return []
    with tsv_path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _baseline_metrics(rows: list[dict]) -> tuple[float | None, float | None]:
    for r in rows:
        if r.get("experiment_id") == BASELINE_ID:
            try:
                return float(r["mean_lddt"]), float(r["total_wall_seconds"])
            except (KeyError, ValueError):
                return None, None
    return None, None


def upsert_experiment_row(
    *,
    experiment_id: str,
    description: str,
    mean_lddt: float,
    median_lddt: float,
    total_wall_seconds: float,
    notes: str = "",
    tsv_path: Path = _DEFAULT_TSV,
) -> Path:
    """Insert or replace the row matching ``experiment_id``."""
    rows = _load(tsv_path)
    baseline_mean, baseline_wall = _baseline_metrics(rows)

    if experiment_id == BASELINE_ID:
        ratio = 1.0
        delta = 0.0
    else:
        ratio = (
            total_wall_seconds / baseline_wall
            if (baseline_wall and baseline_wall > 0) else float("nan")
        )
        delta = (
            (mean_lddt - baseline_mean) / baseline_mean * 100.0
            if (baseline_mean and baseline_mean > 0) else float("nan")
        )

    new_row = {
        "experiment_id": experiment_id,
        "description": description,
        "mean_lddt": f"{mean_lddt:.4f}",
        "median_lddt": f"{median_lddt:.4f}",
        "total_wall_seconds": f"{total_wall_seconds:.1f}",
        "runtime_ratio_vs_baseline": f"{ratio:.3f}",
        "mean_lddt_delta_pct": f"{delta:+.2f}",
        "notes": notes,
    }

    rows = [r for r in rows if r.get("experiment_id") != experiment_id]
    rows.append(new_row)

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return tsv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--mean-lddt", type=float, required=True)
    parser.add_argument("--median-lddt", type=float, required=True)
    parser.add_argument("--total-wall-seconds", type=float, required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--tsv", type=Path, default=_DEFAULT_TSV)
    args = parser.parse_args()
    path = upsert_experiment_row(
        experiment_id=args.experiment_id,
        description=args.description,
        mean_lddt=args.mean_lddt,
        median_lddt=args.median_lddt,
        total_wall_seconds=args.total_wall_seconds,
        notes=args.notes,
        tsv_path=args.tsv,
    )
    print(f"Appended row for {args.experiment_id} to {path}.")


if __name__ == "__main__":
    main()
