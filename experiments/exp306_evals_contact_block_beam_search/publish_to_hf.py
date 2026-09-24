#!/usr/bin/env python
"""Publish exp306 tables, figures, and raw rollouts to the public HF bucket."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/contact-block-beam-exp306"
EXPECTED_SEALED_SHA256 = {
    "sealed_beam4.csv": "e5c89eff477b2c9919d8448ee22ea095377328d006fe89b49d57758762a1c1fb",
    "dev_sealed_beam4.csv": "313194b7d091ca64af8932000b75017a1890cd34d91285e7b45664bdf83d4510",
    "sealed_beam8.csv": "74d863c46c2fbba3900b7d43562ddf506e6cc21eb17a90dde6138efdb432b84a",
}


def main() -> None:
    """Sync reproducible small outputs and auditable per-rollout parquets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="fetch working S3 results and rebuild scored tables and plots")
    args = parser.parse_args()
    if args.refresh:
        fetches = (
            ("val-b4", "beam4", "eval-val", "beam4/eval-val"),
            ("foldswitch-b4", "beam4", "foldswitch", "beam4/foldswitch"),
            ("dev-b4", "beam4", "foldswitch", "dev-beam4/foldswitch"),
            ("dev-b8", "beam8", "foldswitch", "beam8/foldswitch"),
            ("pilot-b8-val", "beam8", "eval-val", "beam8/eval-val"),
        )
        for source, mode, cohort, target in fetches:
            subprocess.run([
                sys.executable, str(HERE / "fetch_results.py"),
                "--source", f"s3://marin-us-east-02a/MarinFold/exp306/{source}",
                "--mode", mode, "--cohort", cohort,
                "--destination", str(HERE / "_cache" / target),
            ], check=True)
        commands = (
            [sys.executable, str(HERE / "score_eval_val.py"), "--mode", "beam4"],
            [sys.executable, str(HERE / "score_foldswitch.py"), "seal", "--mode", "beam4"],
            [sys.executable, str(HERE / "score_foldswitch.py"), "score", "--mode", "beam4"],
            [sys.executable, str(HERE / "score_foldswitch.py"), "seal", "--mode", "beam4",
             "--allow-partial", "--raw-root", str(HERE / "_cache" / "dev-beam4" / "foldswitch"),
             "--output-prefix", "dev_"],
            [sys.executable, str(HERE / "score_foldswitch.py"), "score", "--mode", "beam4",
             "--allow-partial", "--raw-root", str(HERE / "_cache" / "dev-beam4" / "foldswitch"),
             "--output-prefix", "dev_"],
            [sys.executable, str(HERE / "score_foldswitch.py"), "seal", "--mode", "beam8",
             "--allow-partial"],
            [sys.executable, str(HERE / "score_foldswitch.py"), "score", "--mode", "beam8",
             "--allow-partial"],
            [sys.executable, str(HERE / "collect_timings.py")],
            [sys.executable, str(HERE / "plot_results.py")],
            [sys.executable, str(HERE / "build_summary.py")],
        )
        for command in commands:
            subprocess.run(command, cwd=HERE, check=True)
        for name, expected in EXPECTED_SEALED_SHA256.items():
            sealed = HERE / "data" / name
            if hashlib.sha256(sealed.read_bytes()).hexdigest() != expected:
                raise ValueError(f"regenerated {name} differs from frozen result")
    sources = (
        (HERE / "data", "tables"),
        (HERE / "plots", "plots"),
        (HERE / "_cache" / "beam4" / "eval-val", "raw/beam4/eval-val"),
        (HERE / "_cache" / "beam4" / "foldswitch", "raw/beam4/foldswitch"),
        (HERE / "_cache" / "dev-beam4" / "foldswitch", "raw/beam4/foldswitch-dev"),
        (HERE / "_cache" / "beam8" / "foldswitch", "raw/beam8/foldswitch-dev"),
        (HERE / "_cache" / "beam8" / "eval-val", "raw/beam8/eval-val-pilot"),
    )
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
