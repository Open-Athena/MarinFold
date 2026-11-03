"""Restore only this pilot's public inputs and saved predictions, without auth."""

import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

from publish_to_hf import DEST

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Fetch verified archives so scoring/figures need no GPU or credentials."""
    manifest = json.loads((HERE / "data/public_artifacts.json").read_text())
    fs = HfFileSystem(token=False)
    for filename, output in (
        ("inputs.tar.gz", HERE / "inputs"),
        ("structures.tar.gz", HERE / "scratch/results"),
    ):
        path = HERE / "scratch" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = fs.read_bytes(f"{DEST.removeprefix('hf://')}/{filename}")
        if hashlib.sha256(data).hexdigest() != manifest["files"][filename]["sha256"]:
            raise ValueError(f"public archive digest mismatch: {filename}")
        path.write_bytes(data)
        with tarfile.open(path) as archive:
            archive.extractall(output, filter="data")
    print("Restored ten targets and 200 saved structures")


if __name__ == "__main__":
    main()
