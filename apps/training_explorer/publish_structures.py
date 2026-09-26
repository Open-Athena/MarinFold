"""Publish and verify compact structure previews in the shared HF data bucket."""

import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STRUCTURES = HERE / "structures"
PREFIX = "data/training-explorer/2026-09-22/structures"
BUCKET = "open-athena/MarinFold"
BUCKET_URI = f"hf://buckets/{BUCKET}/{PREFIX}"
RESOLVE = f"https://huggingface.co/buckets/{BUCKET}/resolve/"


def main() -> None:
    """Upload every local PDB and write checked public URLs into snapshots."""
    snapshots = {
        name: json.loads((DATA / f"{name}.json").read_text())
        for name in ("latest", "original", "eval")
    }
    manifest = []
    for snapshot in snapshots.values():
        for protein in snapshot["proteins"]:
            filename = hashlib.sha256(protein["id"].encode()).hexdigest()[:20] + ".pdb"
            path = STRUCTURES / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            content = path.read_bytes()
            manifest.append(
                {
                    "proteinId": protein["id"],
                    "file": filename,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    if len(manifest) != 316 or len({item["file"] for item in manifest}) != 316:
        raise ValueError("Expected 316 unique structure previews")
    subprocess.run(
        ["hf", "buckets", "sync", str(STRUCTURES), BUCKET_URI, "-q"], check=True
    )
    listing = subprocess.run(
        ["hf", "buckets", "list", f"{BUCKET}/{PREFIX}", "-R", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = {
        Path(item["path"]).name: item["size"] for item in json.loads(listing.stdout)
    }
    local = {item["file"]: item["bytes"] for item in manifest}
    if remote != local:
        raise ValueError(
            f"Bucket mismatch: {len(remote)} remote previews, {len(local)} local previews"
        )
    for snapshot in snapshots.values():
        snapshot["structureStorage"] = BUCKET_URI
        for protein in snapshot["proteins"]:
            filename = hashlib.sha256(protein["id"].encode()).hexdigest()[:20] + ".pdb"
            url = RESOLVE + quote(f"{PREFIX}/{filename}", safe="")
            protein["structureUrl"] = url
            if "pdbFallbackUrl" in protein:
                protein["pdbFallbackUrl"] = url
    (DATA / "structure_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"))
    )
    for name, snapshot in snapshots.items():
        (DATA / f"{name}.json").write_text(json.dumps(snapshot, separators=(",", ":")))
    print(
        f"Published and verified {len(manifest)} structure previews under {BUCKET_URI}",
        flush=True,
    )


if __name__ == "__main__":
    main()
