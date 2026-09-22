#!/usr/bin/env python
"""Publish exp308 tables, plots, and raw rollouts to the public HF bucket."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/contact-novelty-search-exp308"
PILOT_SOURCE = "s3://marin-us-east-02a/MarinFold/exp308/pilot-v1"
TEST_SOURCE = "s3://marin-us-east-02a/MarinFold/exp308/test-v1"
MODES = (
    "b16_e0p0_d0_w10",
    "b16_e0p05_d0_w10",
    "b16_e0p05_d20_w10",
    "b32_e0p05_d20_w10",
    "b16_e0p2_d20_w10",
)


def run(*parts: str) -> None:
    """Run a reproducibility step and stop on failure."""
    subprocess.run(parts, cwd=HERE, check=True)


def verify_seal(selection: str, mode: str) -> None:
    """Ensure refresh does not silently change a reference-blind shortlist."""
    path = HERE / "data" / f"sealed_{selection}_{mode}.csv"
    expected = path.with_suffix(".sha256").read_text().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"sealed shortlist hash mismatch: {path.name}")


def main() -> None:
    """Optionally rebuild outputs, then sync small and raw public artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    choice = (HERE / "data" / "frozen_choice.txt").read_text().strip()
    if choice not in MODES:
        raise ValueError(f"unexpected frozen choice: {choice}")
    if args.refresh:
        for mode in MODES:
            run(sys.executable, "fetch_results.py", "--source", PILOT_SOURCE, "--mode", mode)
            run(sys.executable, "score_foldswitch.py", "seal", "--mode", mode,
                "--selection", "pilot")
            verify_seal("pilot", mode)
            run(sys.executable, "score_foldswitch.py", "score", "--mode", mode,
                "--selection", "pilot")
        run(sys.executable, "summarize_pilot.py")
        run(sys.executable, "fetch_results.py", "--source", TEST_SOURCE, "--mode", choice)
        run(sys.executable, "score_foldswitch.py", "seal", "--mode", choice,
            "--selection", "test")
        verify_seal("test", choice)
        run(sys.executable, "score_foldswitch.py", "score", "--mode", choice,
            "--selection", "test")
        run(sys.executable, "collect_timings.py")
        run(sys.executable, "plot_results.py")
        run(sys.executable, "build_summary.py")
    sources = [(HERE / "data", "tables"), (HERE / "plots", "plots")]
    sources.extend((HERE / "_cache" / mode / "foldswitch", f"raw/{mode}/foldswitch")
                   for mode in MODES)
    for source, suffix in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        command = ["hf", "buckets", "sync", str(source), f"{DEST}/{suffix}"]
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
