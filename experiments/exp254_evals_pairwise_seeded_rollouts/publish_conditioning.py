# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify and archive the complete controlled-intervention run for public access."""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from verify_conditioning import verify


def main() -> None:
    """Create an archive compatible with publish_to_hf.py publish/fetch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accept-budget-termination", action="store_true")
    parser.add_argument("--worker-sha256", action="append")
    args = parser.parse_args()
    bundle_manifest = json.loads((args.bundle / "bundle_manifest.json").read_text())
    files = {}
    for name, expected in bundle_manifest.items():
        path = args.bundle / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen bundle changed: {name}")
        files[f"inputs/{name}"] = path
    files["inputs/bundle_manifest.json"] = args.bundle / "bundle_manifest.json"
    verification = verify(
        args.bundle / "plan.json",
        args.run,
        accept_budget_termination=args.accept_budget_termination,
        worker_sha256=args.worker_sha256,
    )
    for path in sorted((args.run / "units").iterdir()):
        files[f"run/units/{path.name}"] = path
    for path in sorted(args.run.glob("staging-*.json")):
        files[f"run/{path.name}"] = path
    for path in sorted((args.run / "failures").glob("*.json.gz")):
        files[f"run/initial_failures/{path.name}"] = path
    execution = args.run / "execution.json"
    if execution.exists():
        files["run/execution.json"] = execution
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        verified = Path(temporary) / "verification.json"
        verified.write_text(json.dumps(verification, indent=2) + "\n")
        files["verification.json"] = verified
        records = {
            name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for name, path in files.items()
        }
        with tarfile.open(args.out, "w:gz") as archive:
            for name, path in sorted(files.items()):
                archive.add(path, arcname=name, recursive=False)
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "experiment": 254,
        "scope": "97 eval-val proteins; controlled contact-prompt intervention; 155200 new completions",
        "plan_sha256": verification["plan_sha256"],
        "archive": {
            "uri": f"hf://buckets/open-athena/MarinFold/data/exp254/conditioning-v1/{sha[:16]}/{args.out.name}",
            "sha256": sha,
            "bytes": args.out.stat().st_size,
        },
        "files": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["archive"], indent=2))


if __name__ == "__main__":
    main()
