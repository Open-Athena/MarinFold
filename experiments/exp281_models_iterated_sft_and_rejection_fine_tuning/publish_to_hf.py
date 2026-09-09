"""Publish a deterministic, bounded archive of the collected S3 trial report.

The input is the report.py W&B artifact directory, including its S3 URI index.
Only indexed files are published; checkpoints and unrelated local files are
excluded. Requires an authenticated, open-athena-scoped `hf` CLI on PATH.
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", default=(
        "hf://buckets/open-athena/MarinFold/data/exp281/trial-s03/report.zip"))
    args = parser.parse_args()
    index = json.loads((args.report / "index.json").read_text())
    names = sorted([*index, "index.json"])
    if any(Path(name).name != name for name in names):
        raise ValueError("report index must contain simple filenames")
    total = sum((args.report / name).stat().st_size for name in names)
    if total > 10_000_000:
        raise ValueError(f"report exceeds 10 MB budget: {total}")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w") as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (args.report / name).read_bytes())
    subprocess.run(["hf", "buckets", "cp", str(args.archive), args.destination], check=True)
    print(json.dumps({"destination": args.destination, "files": len(names),
                      "bytes": args.archive.stat().st_size,
                      "sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
