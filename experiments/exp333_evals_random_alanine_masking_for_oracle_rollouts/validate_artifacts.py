"""Validate completeness and mask invariants before reading accuracy."""

import argparse
import json
from pathlib import Path

import pandas as pd

from masking_policy import mutation_count

HERE = Path(__file__).resolve().parent
TARGETS = (
    HERE.parent
    / "exp321_evals_null_sequence_contrastive_guidance_for_contacts"
    / "data"
    / "targets.csv"
)


def mode_fraction(mode: str) -> float:
    """Decode the four-digit per-mille suffix used in raw mode names."""
    return int(mode.removeprefix("mask_p")) / 1000


def main() -> None:
    """Fail on incomplete outputs, non-nested masks, or capped generations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    modes.sort(key=mode_fraction)
    targets = pd.read_csv(TARGETS)
    targets = targets[(targets.cohort == "eval-val") & (targets.split == args.split)]
    expected_stems = set(targets.stem)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary = {
        "split": args.split,
        "modes": modes,
        "expected_proteins": len(expected_stems),
        "expected_rollouts_per_protein": args.n_rollouts,
        "raw_files": 0,
        "timing_files": 0,
        "rollouts": 0,
        "unfinished": 0,
        "malformed_contact_statements": 0,
        "out_of_range_contact_statements": 0,
        "max_n_tokens_minus_budget": 0,
    }
    for mode in modes:
        root = HERE / "_cache" / mode / "eval-val"
        raw_paths = sorted(path for path in root.glob("*.parquet") if ".timing." not in path.name)
        timing_paths = sorted(root.glob("*.timing.parquet"))
        raw_stems = {path.stem for path in raw_paths}
        timing_stems = {path.name.removesuffix(".timing.parquet") for path in timing_paths}
        if raw_stems != expected_stems or timing_stems != expected_stems:
            raise ValueError(
                f"{mode}: raw missing={sorted(expected_stems - raw_stems)} extra={sorted(raw_stems - expected_stems)}; "
                f"timing missing={sorted(expected_stems - timing_stems)} extra={sorted(timing_stems - expected_stems)}"
            )
        summary["raw_files"] += len(raw_paths)
        summary["timing_files"] += len(timing_paths)
        for path in raw_paths:
            frame = pd.read_parquet(path).sort_values("rollout")
            stem = path.stem
            if list(frame.rollout.astype(int)) != list(range(args.n_rollouts)):
                raise ValueError(f"{mode}/{stem}: rollout IDs are incomplete")
            fraction = mode_fraction(mode)
            if not (frame.mutation_fraction.astype(float) == fraction).all():
                raise ValueError(f"{mode}/{stem}: mutation fraction mismatch")
            expected_count = mutation_count(int(frame.n_mutable.iloc[0]), fraction)
            if not (frame.n_mutated.astype(int) == expected_count).all():
                raise ValueError(f"{mode}/{stem}: mutation count mismatch")
            length = int(frame.L.iloc[0])
            for contacts in frame.contacts:
                if any(not (0 <= int(left) < int(right) < length) for left, right in contacts):
                    raise ValueError(f"{mode}/{stem}: invalid contact coordinates")
            frames[(mode, stem)] = frame
            summary["rollouts"] += len(frame)
            summary["unfinished"] += int((~frame.finished.astype(bool)).sum())
            summary["malformed_contact_statements"] += int(frame.malformed_contacts.sum())
            summary["out_of_range_contact_statements"] += int(frame.out_of_range_contacts.sum())
            summary["max_n_tokens_minus_budget"] = max(
                int(summary["max_n_tokens_minus_budget"]),
                int((frame.n_tokens - frame.max_new).max()),
            )
    for stem in expected_stems:
        for rollout in range(args.n_rollouts):
            previous: set[int] = set()
            for mode in modes:
                positions = {
                    int(value)
                    for value in frames[(mode, stem)].iloc[rollout].mutated_positions
                }
                if not previous <= positions:
                    raise ValueError(f"{stem} rollout {rollout}: masks are not nested at {mode}")
                previous = positions
    if summary["unfinished"]:
        raise ValueError(f"{summary['unfinished']} rollouts hit the token cap")
    if summary["max_n_tokens_minus_budget"] > 0:
        raise ValueError("one or more rollouts exceeded its token budget")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
