"""Publish Neff figures, exact frozen alignments, counts and source-traced tables."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/neff-2026-10-13"


def digest(path: Path) -> str:
    """Hash an artifact without retaining its bytes in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Build a reproducible public bundle and optionally upload it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "data/neff_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(HERE / "data" / name) != expected:
            raise ValueError(f"Stale Neff result: {name}")
    inputs = HERE / "scratch/alphafold/inputs"
    if digest(inputs / "targets.json") != manifest["msa_inputs_sha256"]:
        raise ValueError("Frozen input manifest changed")
    stage = HERE / "scratch/public_neff"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for pattern in ("data/*.csv", "data/*.json", "data/inputs/Lato-*", "plots/*", "site/*",
                    "*.py", "*.md", "pyproject.toml", "uv.lock", "generation/*neff*.py"):
        for source in HERE.glob(pattern):
            if not source.is_file() or source.name == "neff_publication.json":
                continue
            target = stage / source.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    # Only frozen sequence/alignment inputs, never predictor weights or caches.
    with tarfile.open(stage / "neff_alignment_inputs.tar", "w") as archive:
        archive.add(inputs / "targets.json", arcname="inputs/targets.json")
        for record in json.loads((inputs / "targets.json").read_text()):
            path = inputs / f"{record['stem']}.a3m.gz"
            if digest(path) != record["msa_sha256"]:
                raise ValueError(f"Alignment changed: {record['stem']}")
            archive.add(path, arcname="inputs/" + path.name)
    with tarfile.open(stage / "neff_neighbor_counts.tar.gz", "w:gz") as archive:
        archive.add(HERE / "scratch/neff/counts", arcname="counts")
    for relative, pin in manifest["sources"].items():
        source = HERE.parents[1] / relative
        if digest(source) != pin["sha256"]:
            raise ValueError(f"Neff source changed: {relative}")
        target = stage / "source_tables" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    reference = HERE.parent / "exp74_evals_protenix_pyconfind_contacts/msa_depth.py"
    shutil.copyfile(reference, stage / "reference_msa_depth.py")
    inventory = {str(path.relative_to(stage)): dict(bytes=path.stat().st_size, sha256=digest(path))
                 for path in sorted(stage.rglob("*")) if path.is_file()}
    publication = dict(destination=DESTINATION, files=inventory,
                       alignment_inputs="All 333 original predictor A3Ms plus their frozen query/hash manifest",
                       neighbor_counts="Per-original-record Neff80 neighbor counts and measured worker metadata")
    text = json.dumps(publication, indent=2) + "\n"
    (stage / "publication_manifest.json").write_text(text)
    (HERE / "data/neff_publication.json").write_text(text)
    print(f"{len(inventory)} files, {sum(r['bytes'] for r in inventory.values()) / 1e6:.1f} MB → {DESTINATION}", flush=True)
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), DESTINATION, "--quiet"], check=True)


if __name__ == "__main__":
    main()
