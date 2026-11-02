"""Restore only the public 116-protein evaluation inputs for reproducing this run."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Read inputs anonymously, verifying the publication manifest."""
    plan = json.loads((HERE / "data/plan.json").read_text())
    root = plan["artifact_root"].removeprefix("hf://")
    fs = HfFileSystem(token=False)
    manifest = json.loads(fs.read_text(root + "/data/publication_manifest.json"))
    for record in manifest["files"]:
        if not record["path"].startswith("inputs/"):
            continue
        content = fs.read_bytes(root + "/" + record["path"])
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        path = HERE / record["path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        print("RESTORED", path.name)


if __name__ == "__main__":
    main()
