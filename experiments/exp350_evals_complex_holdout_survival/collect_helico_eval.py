"""Validate Helico runs and collect timings plus an evaluation manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from score_helico_structures import parse_run_name

HERE = Path(__file__).resolve().parent
DEFAULT_PER_TARGET = HERE / "data/helico_per_target.csv"
DEFAULT_SUMMARY = HERE / "data/helico_summary.csv"
DEFAULT_COMPARISONS = HERE / "data/helico_comparisons.csv"
DEFAULT_TIMINGS = HERE / "data/helico_timings.csv"
DEFAULT_MANIFEST = HERE / "data/helico_eval_manifest.json"
HELICO_COMMIT = "75f7baf4e9147cf97ca212c3a71b4d367ef198d9"
PUBLIC_PREFIX = (
    "hf://buckets/open-athena/MarinFold/data/evals/"
    "exp350_foldbench_pair_holdout/contact_eval_v1/helico_v1"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows with stable field order."""
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_timing(row: dict[str, str], split: str, arm: str, run_name: str) -> dict:
    """Normalize early runs whose total omitted the separately timed model load."""
    elapsed = float(row["elapsed_seconds"])
    model_load = float(row["model_load_seconds"])
    total = float(row["total_seconds"])
    timing_contract_fixed = total + 1e-9 >= elapsed + model_load
    if not timing_contract_fixed:
        total += model_load
    return {
        "split": split,
        "arm": arm,
        "run_name": run_name,
        **row,
        "total_seconds": total,
        "total_seconds_normalized": not timing_contract_fixed,
    }


def main() -> None:
    """Collect complete run metadata after structural scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--per-target", type=Path, default=DEFAULT_PER_TARGET)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    score_rows = read_csv(args.per_target)
    summary_rows = read_csv(args.summary)
    scored_runs = {row["run_name"] for row in score_rows}
    timing_rows = []
    run_records = []
    for run_dir in sorted(args.run_dir):
        split, arm = parse_run_name(run_dir)
        if run_dir.name not in scored_runs:
            raise ValueError(f"No pair-specific scores for {run_dir.name}")
        meta_path = run_dir / "meta.json"
        timings_path = run_dir / "timings.csv"
        predictions = sorted((run_dir / "predictions").glob("*.pkl"))
        if not meta_path.exists() or not timings_path.exists() or not predictions:
            raise ValueError(f"Incomplete Helico run cache: {run_dir}")
        meta = json.loads(meta_path.read_text())
        expected = 6 if split == "dev" else 17
        rows = read_csv(timings_path)
        if len(rows) != expected or len(predictions) != expected:
            raise ValueError(
                f"{run_dir.name}: expected {expected} timings/predictions, "
                f"got {len(rows)}/{len(predictions)}"
            )
        timing_rows.extend(
            normalized_timing(row, split, arm, run_dir.name) for row in rows
        )
        run_records.append(
            {
                "run_name": run_dir.name,
                "split": split,
                "arm": arm,
                "targets": expected,
                "modal_volume_path": (
                    "/experiments/exp350_marinfold_complex_eval/" + run_dir.name
                ),
                "meta": meta,
                "artifacts": {
                    str(path.relative_to(run_dir)): sha256(path)
                    for path in sorted(run_dir.rglob("*"))
                    if path.is_file()
                },
            }
        )

    required_dev = {
        f"marinfold_{scope}_{budget}"
        for scope in ("all", "intra")
        for budget in ("L5", "L2", "L")
    }
    required_test = {"off", "oracle", "marinfold_all_L", "marinfold_intra_L"}
    observed_dev = {row["arm"] for row in summary_rows if row["split"] == "dev"}
    observed_test = {row["arm"] for row in summary_rows if row["split"] == "test"}
    if observed_dev != required_dev or observed_test != required_test:
        raise ValueError(
            f"Structural arm mismatch: dev={sorted(observed_dev)}, "
            f"test={sorted(observed_test)}"
        )
    all_dev = [
        row
        for row in summary_rows
        if row["split"] == "dev" and row["arm"].startswith("marinfold_all_")
    ]
    selected = max(all_dev, key=lambda row: float(row["mean_dockq"]))
    if selected["arm"] != "marinfold_all_L":
        raise ValueError(f"Unexpected development-selected arm: {selected}")

    timing_rows.sort(key=lambda row: (row["split"], row["arm"], row["stem"]))
    write_csv(args.timings, timing_rows)
    manifest = {
        "dataset": "foldbench_complex_pair_holdout_contact_eval_v1",
        "targets_sha256": sha256(
            HERE / "data/foldbench_complex_contact_eval_targets.parquet"
        ),
        "helico_git_commit": HELICO_COMMIT,
        "helico_checkpoint": "/ckpts/contacts-msafree-01/final.pt",
        "helico_checkpoint_step": 6000,
        "inference": {
            "single_sequence": True,
            "n_seeds": 1,
            "n_samples_per_seed": 3,
            "n_cycles": 6,
            "max_tokens": 2048,
            "sample_selection": "maximum Helico ranking_score; no ground-truth selection",
        },
        "scoring": {
            "dockq_version": "2.1.3",
            "pair_specific": True,
            "homodimer_mapping": "maximum over the two chain permutations",
            "success_threshold": 0.23,
            "group_bootstrap_replicates": 10_000,
            "group_bootstrap_seed": 350,
        },
        "development_selection": {
            "criterion": "highest mean pair-specific DockQ over all-contact budgets",
            "selected_arm": selected["arm"],
            "mean_dockq": float(selected["mean_dockq"]),
        },
        "contact_arms": {
            path.name: sha256(path)
            for path in sorted((HERE / "data/helico_arms").iterdir())
            if path.is_file()
        },
        "public_prefix": PUBLIC_PREFIX,
        "derived": {
            args.per_target.name: sha256(args.per_target),
            args.summary.name: sha256(args.summary),
            args.comparisons.name: sha256(args.comparisons),
            args.timings.name: sha256(args.timings),
        },
        "runs": run_records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"collected {len(run_records)} runs and {len(timing_rows)} timings; "
        f"selected {selected['arm']}"
    )


if __name__ == "__main__":
    main()
