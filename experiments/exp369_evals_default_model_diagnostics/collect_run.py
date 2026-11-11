"""Collect small prediction artifacts from CoreWeave with a checked transfer budget."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import s3fs


def collect_one(fs: s3fs.S3FileSystem, remote: str, target: Path, size: int) -> int:
    """Download one artifact atomically, retaining completed local downloads."""
    if target.exists() and target.stat().st_size == size:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = fs.cat_file(remote)
    if len(payload) != size:
        raise ValueError(f"Object changed while downloading: {remote}")
    temporary = target.with_name(target.name + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return size


def main() -> None:
    """Inventory before downloading; default to votes, timings, and validation markers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if not 0 < args.max_bytes <= 10_000_000_000:
        raise ValueError("Transfer budget must be positive and at most 10 GB")
    fs = s3fs.S3FileSystem(profile="cw", client_kwargs={"endpoint_url": "https://cwobject.com"},
                          config_kwargs={"s3": {"addressing_style": "virtual"}})
    root = args.run.removeprefix("s3://").rstrip("/")
    objects = fs.find(root, detail=True)
    suffixes = (".npz", ".complete.json", ".timing.json") + ((".json.gz",) if args.include_raw else ())
    selected = [(name, item) for name, item in objects.items() if name.endswith(suffixes)]
    total = sum(item["size"] for _, item in selected)
    print(json.dumps({"objects": len(selected), "total_bytes": total, "include_raw": args.include_raw}), flush=True)
    if total > args.max_bytes:
        raise ValueError("Artifact selection exceeds the declared transfer budget")
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(collect_one, fs, name, args.out / name.removeprefix(root + "/"), item["size"])
                   for name, item in selected]
        downloaded = sum(future.result() for future in futures)
    manifest = {"source": args.run, "total_bytes": total, "downloaded_bytes": downloaded,
                "files": [{"path": name.removeprefix(root + "/"), "size": item["size"],
                           "sha256": hashlib.sha256((args.out / name.removeprefix(root + "/")).read_bytes()).hexdigest()}
                          for name, item in selected]}
    (args.out / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Collected {len(selected)} objects; downloaded {downloaded} bytes", flush=True)


if __name__ == "__main__":
    main()
