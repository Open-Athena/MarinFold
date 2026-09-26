"""Attach published source-backbone URLs to every displayed sequence hit."""

import argparse
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data%2Ftraining-explorer%2F2026-09-22%2Fneighbor-structures%2Fmanifest.json"
)


def load_manifest(path: Path | None) -> dict:
    """Read a local manifest or the anonymously accessible published copy."""
    if path is not None:
        return json.loads(path.read_text())
    with urllib.request.urlopen(MANIFEST_URL, timeout=120) as response:
        return json.load(response)


def note(hit_id: str) -> str:
    """Describe how the displayed backbone relates to the sequence hit."""
    arm = hit_id.split("|", 1)[0]
    if arm == "mpnn-afdb":
        return "AlphaFold DB source backbone used for this synthetic redesign"
    if arm == "mpnn-esm":
        return "ESMFold2 source backbone used for this synthetic redesign"
    if arm == "afdb":
        return "AlphaFold DB v4 training source backbone"
    if arm == "esm_atlas":
        return "Exact ESMFold2 source backbone"
    raise ValueError(f"Unknown neighbor source arm: {arm}")


def main() -> None:
    """Validate complete hit coverage and update all three browser snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    sources = {row["sourceKey"]: row for row in manifest["sources"]}
    hit_sources = manifest["hitSources"]
    seen = set()
    for name in ("latest", "original", "eval"):
        path = DATA / f"{name}.json"
        snapshot = json.loads(path.read_text())
        snapshot["neighborStructureStorage"] = manifest["publicPrefix"]
        for protein in snapshot["proteins"]:
            for hit in protein.get("neighbors", []):
                source_key = hit_sources.get(hit["id"])
                if source_key is None:
                    raise ValueError(f"Manifest lacks hit {hit['id']}")
                source = sources.get(source_key)
                if source is None:
                    raise ValueError(f"Manifest lacks source {source_key}")
                hit["structureSourceKey"] = source_key
                hit["structureUrl"] = source["url"]
                hit["structureFormat"] = "pdb"
                hit["structureNote"] = note(hit["id"])
                seen.add(hit["id"])
        path.write_text(json.dumps(snapshot, separators=(",", ":")))
    if seen != set(hit_sources):
        raise ValueError(
            f"Manifest has {len(set(hit_sources) - seen)} hit IDs absent from snapshots"
        )
    (DATA / "neighbor_structure_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"))
    )
    print(
        f"Attached {len(seen)} unique hits to {len(sources)} source backbones",
        flush=True,
    )


if __name__ == "__main__":
    main()
