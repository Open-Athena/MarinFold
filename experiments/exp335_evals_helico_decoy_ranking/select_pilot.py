"""Select a deterministic target-balanced Helico pilot from the full benchmark."""

import argparse
import csv
from pathlib import Path

from benchmark import af2rank_composite, load_af2rank_rows, rankdata


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a non-empty CSV file."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV with stable columns."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile_ranks(values: list[float]) -> list[float]:
    """Return tie-aware ranks scaled to the closed interval [0, 1]."""
    ranks = rankdata(values)
    if len(ranks) == 1:
        return [0.5]
    return [(rank - 1) / (len(ranks) - 1) for rank in ranks]


def select_targets(
    rows: list[dict[str, str]], levels: list[float]
) -> list[dict[str, str]]:
    """Choose unique targets nearest a grid of length/count percentiles."""
    length_percentiles = percentile_ranks(
        [float(row["n_native_residues"]) for row in rows]
    )
    count_percentiles = percentile_ranks([float(row["n_decoys"]) for row in rows])
    candidates = [
        (row, length_percentile, count_percentile)
        for row, length_percentile, count_percentile in zip(
            rows, length_percentiles, count_percentiles, strict=True
        )
    ]

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for desired_length in levels:
        for desired_count in levels:
            row, actual_length, actual_count = min(
                (
                    candidate
                    for candidate in candidates
                    if candidate[0]["target"] not in used
                ),
                key=lambda candidate: (
                    (candidate[1] - desired_length) ** 2
                    + (candidate[2] - desired_count) ** 2,
                    candidate[0]["target"],
                ),
            )
            used.add(row["target"])
            selected.append(
                {
                    **row,
                    "desired_length_percentile": desired_length,
                    "desired_decoy_count_percentile": desired_count,
                    "actual_length_percentile": actual_length,
                    "actual_decoy_count_percentile": actual_count,
                }
            )
    return selected


def quantile_indices(size: int, count: int) -> list[int]:
    """Select unique, evenly spaced indices including both endpoints."""
    if count < 2:
        raise ValueError("count must be at least two")
    if size < count:
        raise ValueError(f"cannot choose {count} unique rows from {size}")
    indices = [round(index * (size - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("quantile rounding produced duplicate indices")
    return indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-summary", type=Path, default=Path("data/target_summary.csv")
    )
    parser.add_argument("--af2rank-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--decoys-per-target", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    """Write pilot targets and TM-stratified candidate identifiers."""
    args = parse_args()
    target_summary = load_csv(args.target_summary)
    selected_targets = select_targets(target_summary, levels=[0.1, 0.5, 0.9])
    selected_ids = {row["target"] for row in selected_targets}

    af2rank_rows = load_af2rank_rows(args.af2rank_csv)
    by_target: dict[str, list[dict[str, str]]] = {target: [] for target in selected_ids}
    native_by_target: dict[str, dict[str, str]] = {}
    for row in af2rank_rows:
        target = row["target"]
        if target not in selected_ids:
            continue
        if row["decoy_id"] == "native":
            native_by_target[target] = row
        elif row["decoy_id"] != "none":
            by_target[target].append(row)

    candidate_rows: list[dict[str, str | float | int]] = []
    for target in sorted(selected_ids):
        native = native_by_target[target]
        candidate_rows.append(
            {
                "target": target,
                "decoy_id": "native",
                "candidate_kind": "native",
                "tm_quantile": 1.0,
                "tmscore": float(native["tmscore"]),
                "gdt_ts": float(native["gdt_ts"]),
                "af2rank_composite": af2rank_composite(native),
                "af2_ptm": float(native["ptm"]),
                "deepaccnet": float(native["danscore"]),
                "negative_rosetta_energy": -float(native["rosettascore"]),
            }
        )
        decoys = sorted(
            by_target[target], key=lambda row: (float(row["tmscore"]), row["decoy_id"])
        )
        indices = quantile_indices(len(decoys), args.decoys_per_target)
        for quantile_number, decoy_index in enumerate(indices):
            row = decoys[decoy_index]
            candidate_rows.append(
                {
                    "target": target,
                    "decoy_id": row["decoy_id"],
                    "candidate_kind": "decoy",
                    "tm_quantile": quantile_number / (args.decoys_per_target - 1),
                    "tmscore": float(row["tmscore"]),
                    "gdt_ts": float(row["gdt_ts"]),
                    "af2rank_composite": af2rank_composite(row),
                    "af2_ptm": float(row["ptm"]),
                    "deepaccnet": float(row["danscore"]),
                    "negative_rosetta_energy": -float(row["rosettascore"]),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pilot_targets.csv", selected_targets)
    write_csv(args.output_dir / "pilot_candidates.csv", candidate_rows)
    print(
        f"selected {len(selected_targets)} targets and {len(candidate_rows)} candidates "
        f"({args.decoys_per_target} decoys + native per target)"
    )


if __name__ == "__main__":
    main()
