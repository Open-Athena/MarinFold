"""Preserve one completed rolling checkpoint with a same-bucket server-side copy."""

import argparse
import configparser
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import fsspec

from experiments.exp363_models_double_channel_width.config import PREFIX, RUN_ID


def main() -> None:
    """Copy checkpoint objects, verify their sizes, then publish metadata last."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    args = parser.parse_args()
    source = args.source.rstrip("/")
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    fs = fsspec.filesystem(
        "s3", key=cw["aws_access_key_id"], secret=cw["aws_secret_access_key"],
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    with fs.open(f"{source}/metadata.json") as handle:
        metadata = json.load(handle)
    step = int(metadata["step"])
    destination = f"{PREFIX}/fork_sources/{RUN_ID}/step-{step}"
    source_root = fs._strip_protocol(source)
    destination_root = fs._strip_protocol(destination)
    if source_root.split("/", 1)[0] != destination_root.split("/", 1)[0]:
        raise ValueError("Fork preservation requires a same-bucket server-side copy")
    if source_root.rsplit("/", 1)[-1] != f"step-{step}":
        raise ValueError("Checkpoint directory and metadata step disagree")
    if fs.exists(f"{destination}/metadata.json"):
        raise FileExistsError(f"Completed fork source already exists: {destination}")
    objects = fs.find(source, detail=True)
    files = {p: v for p, v in objects.items() if v["type"] == "file"}
    with fs.open(f"{source}/manifest.json") as handle:
        manifest = json.load(handle)
    array_paths = [array["path"] for array in manifest["arrays"]]
    for prefix in ("model/", "opt_state/", "training_key", "step"):
        if not any(p.startswith(prefix) for p in array_paths):
            raise ValueError(f"Incomplete source: missing {prefix}")

    def copy_object(path: str) -> None:
        target = destination_root + path[len(source_root):]
        fs.cp_file(path, target)
        if fs.info(target)["size"] != files[path]["size"]:
            raise ValueError(f"Copied object size mismatch: {target}")

    payload = [p for p in files if p != f"{source_root}/metadata.json"]
    total_bytes = sum(v["size"] for v in files.values())
    print(f"Preserving step {step}: {len(files)} objects, {total_bytes:,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(copy_object, payload))
    copy_object(f"{source_root}/metadata.json")
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "destination": destination, "metadata": metadata,
        "object_count": len(files), "total_bytes": total_bytes,
        "method": "S3 same-bucket server-side copy; each object size verified; completion metadata copied last",
    }
    output = Path(__file__).parent / "data/fork_source.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
