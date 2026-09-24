"""Exactly expand the initially amortized model-load timing fields."""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def parse_count(value: str) -> tuple[str, int]:
    """Parse a COHORT=N invocation-size declaration."""
    cohort, separator, raw_count = value.partition("=")
    try:
        count = int(raw_count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected COHORT=N with N >= 1") from error
    if not separator or not cohort or count < 1:
        raise argparse.ArgumentTypeError("expected COHORT=N with N >= 1")
    return cohort, count


def main() -> None:
    """Repair timing parquets using exact recorded shares and launch sizes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "_cache")
    parser.add_argument(
        "--jobs-per-cohort",
        type=parse_count,
        action="append",
        required=True,
        help="Exact number of target-arm jobs in one model-loading invocation.",
    )
    args = parser.parse_args()
    jobs_per_cohort = dict(args.jobs_per_cohort)
    paths = sorted(args.root.glob("*/*/*.timing.parquet"))
    if not paths:
        raise ValueError(f"no timing parquets under {args.root}")
    observed: dict[str, int] = {}
    frames = {}
    for path in paths:
        frame = pd.read_parquet(path)
        if len(frame) != 1:
            raise ValueError(f"{path}: expected one timing row")
        cohort = str(frame.cohort.iloc[0])
        observed[cohort] = observed.get(cohort, 0) + 1
        frames[path] = frame
    if observed != jobs_per_cohort:
        raise ValueError(
            f"timing-file counts {observed} do not match declared launch sizes "
            f"{jobs_per_cohort}"
        )
    repaired = 0
    for path, frame in frames.items():
        if (
            "model_load_accounting" in frame
            and (frame.model_load_accounting == "full_invocation").all()
        ):
            continue
        cohort = str(frame.cohort.iloc[0])
        launch_size = jobs_per_cohort[cohort]
        amortized = float(frame.model_load_seconds.iloc[0])
        full_load = amortized * launch_size
        frame["model_load_seconds"] = full_load
        frame["total_seconds"] = frame.total_seconds + full_load - amortized
        frame["total_seconds_reconstructed"] = False
        frame["model_load_accounting"] = "full_invocation"
        frame["timing_correction"] = "exact_expand_amortized_model_load"
        frame.to_parquet(path, index=False)
        repaired += 1
    print(
        f"verified {len(paths)} timing parquets; repaired {repaired}: {jobs_per_cohort}"
    )


if __name__ == "__main__":
    main()
