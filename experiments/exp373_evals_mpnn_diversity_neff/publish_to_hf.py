"""Publish the small reproducible report and inputs alongside already-public raw samples."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Upload closed artifacts and verify anonymous readback by SHA-256."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE / "data/plan.json").read_text())
    root = plan["artifact_root"].removeprefix("hf://")
    files = list((HERE / "inputs").glob("*"))
    if not args.inputs_only:
        files += list((HERE / "data").glob("*")) + list((HERE / "plots").glob("*"))
        files += [
            p for p in HERE.glob("*") if p.suffix in [".py", ".md", ".toml", ".lock"]
        ]
    fs = HfFileSystem()
    records = []
    for path in sorted(files):
        if not path.is_file():
            continue
        relative = str(path.relative_to(HERE))
        destination = root + "/" + relative
        fs.put_file(str(path), destination)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Fresh anonymous filesystem prevents stale listings after upload.
        content = HfFileSystem(token=False, skip_instance_cache=True).read_bytes(
            destination
        )
        assert hashlib.sha256(content).hexdigest() == digest
        records.append({"path": relative, "bytes": len(content), "sha256": digest})
        print("VERIFIED", relative, len(content), flush=True)
    manifest = HERE / "data/publication_manifest.json"
    manifest.write_text(
        json.dumps({"root": "hf://" + root, "files": records}, indent=2) + "\n"
    )
    fs.put_file(str(manifest), root + "/data/publication_manifest.json")


if __name__ == "__main__":
    main()
