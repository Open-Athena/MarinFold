"""Publish the frozen FoldBench complex evaluation bundle to the HF bucket."""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = Path("/data/exp350_foldbench_eval/bundle")
HF_PREFIX = (
    "hf://buckets/open-athena/MarinFold/"
    "data/evals/exp350_foldbench_pair_holdout/v1"
)
HF = Path.home() / ".local/bin/hf"


def main() -> None:
    """Validate the manifest and synchronize the complete public bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest["public_prefix"] != HF_PREFIX:
        raise ValueError(
            f"Manifest prefix {manifest['public_prefix']!r} != {HF_PREFIX!r}"
        )
    selection = manifest["selection"]
    if selection["included"] != 30 or selection["dev"] + selection["test"] != 30:
        raise ValueError(f"Unexpected frozen selection: {selection}")
    command = [str(HF), "buckets", "sync", str(args.bundle), HF_PREFIX]
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
