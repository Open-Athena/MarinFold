"""Publish the small reproducible report and inputs alongside already-public raw samples."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfFileSystem
from huggingface_hub.utils import disable_progress_bars

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Upload closed artifacts and verify anonymous readback by SHA-256."""
    disable_progress_bars()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    plan_name = "extension_plan.json" if args.extend else "plan.json"
    plan = json.loads((HERE / "data" / plan_name).read_text())
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
        if (
            not path.is_file()
            or path.name == "publication_manifest_1000.json"
            or (not args.extend and path.name == "publication_manifest.json")
        ):
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
    manifest = (
        HERE
        / "data"
        / (
            "publication_manifest_1000.json"
            if args.extend
            else "publication_manifest.json"
        )
    )
    manifest.write_text(
        json.dumps({"root": "hf://" + root, "files": records}, indent=2) + "\n"
    )
    remote_manifest = root + "/data/" + manifest.name
    fs.put_file(str(manifest), remote_manifest)
    assert (
        HfFileSystem(token=False, skip_instance_cache=True).read_bytes(remote_manifest)
        == manifest.read_bytes()
    )
    print("VERIFIED public manifest", len(records), "files", flush=True)


if __name__ == "__main__":
    main()
