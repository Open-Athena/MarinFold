"""Publish prepared figures and raw predictions to the project public HF bucket.

Requires the completed contact collection and generation/archive_helico.py.
No model weights or credentials are included. Default is a local inventory;
--upload performs the authorized publication to this experiment's own prefix.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/exp277-step266344/v6-ptm-ranking"
SOURCE_SEARCH_DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/esmc-training-source-search-2026-09-30"


def publish_source_search(upload: bool) -> None:
    """Package the small source audit without rebuilding predictor archives."""
    paths = [HERE / name for name in (
        "TRAINING_SOURCE_SEARCH.md", "LOW_DEPTH_TRAINING_AUDIT.md", "summary_narrative.md",
        "pyproject.toml", "uv.lock", "publish_to_hf.py", "plots/summary.pdf",
        "generation/search_training_sources.py", "generation/audit_low_depth_msas.py",
        "generation/test_training_source_search.py")]
    for pattern in ("data/training_source_*.csv", "data/training_source_*.json",
                    "data/low_depth_*audit*.csv", "data/low_depth_msa_hits.csv",
                    "data/low_depth_*manifest.json", "data/inputs/training_source_search/**/*",
                    "data/inputs/low_depth_training_audit/**/*"):
        paths.extend(path for path in HERE.glob(pattern) if path.is_file())
    paths = sorted(set(paths))
    inventory = {str(path.relative_to(HERE)): {
        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths}
    print(f"{len(paths)} files, {sum(row['bytes'] for row in inventory.values()) / 1e6:.1f} MB → {SOURCE_SEARCH_DESTINATION}")
    if not upload:
        return
    stage = HERE / "scratch/public_source_search"
    if stage.exists():
        shutil.rmtree(stage)
    for path in paths:
        destination = stage / path.relative_to(HERE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (stage / "publication_manifest.json").write_text(json.dumps({
        "destination": SOURCE_SEARCH_DESTINATION, "files": inventory}, indent=2) + "\n")
    subprocess.run(["hf", "buckets", "sync", str(stage), SOURCE_SEARCH_DESTINATION, "--quiet"], check=True)


def main() -> None:
    """Build a checksummed public package from frozen local results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--source-search-only", action="store_true")
    args = parser.parse_args()
    if args.source_search_only:
        publish_source_search(args.upload)
        return
    stage = HERE / "scratch/public"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for folder in ("data", "plots", "site"):
        shutil.copytree(HERE / folder, stage / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("publication_manifest.json"))
    for name in ("README.md", "DRAFT.md", "FIGURES.md", "RUNBOOK.md", "summary_narrative.md",
                 "TRAINING_SOURCE_SEARCH.md", "LOW_DEPTH_TRAINING_AUDIT.md",
                 "sources_remote.json", "pyproject.toml", "uv.lock"):
        shutil.copyfile(HERE / name, stage / name)
    for path in HERE.glob("*.py"):
        shutil.copyfile(path, stage / path.name)
    (stage / "generation").mkdir()
    for path in (HERE / "generation").iterdir():
        if path.is_file() and (path.suffix in {".py", ".json", ".toml", ".lock"}):
            shutil.copyfile(path, stage / "generation" / path.name)
    with tarfile.open(stage / "contact_rollouts.tar.gz", "w:gz") as archive:
        archive.add(HERE / "scratch/contacts/results", arcname="results")
        archive.add(HERE / "scratch/contacts/inputs", arcname="inputs")
    with tarfile.open(stage / "helico_inputs.tar.gz", "w:gz") as archive:
        archive.add(HERE / "scratch/helico", arcname="inputs")
    shutil.copyfile(HERE / "scratch/helico_coordinates.tar.gz", stage / "helico_coordinates.tar.gz")
    for name in ("esmfold2_predictions.tar.gz", "helico_coordinates.tar.gz"):
        shutil.copyfile(HERE / "scratch/structured_decoys" / name, stage / f"structured_{name}")
    for variant in ("af2", "af3", "boltz2"):
        raw = HERE / "scratch" / ("boltz2" if variant == "boltz2" else "alphafold")
        shutil.copyfile(raw / f"{variant}_predictions.tar.gz", stage / f"{variant}_predictions.tar.gz")
        shutil.copyfile(raw / f"{variant}_contacts.json", stage / f"{variant}_contacts.json")
    with tarfile.open(stage / "alphafold_inputs.tar.gz", "w:gz") as archive:
        archive.add(HERE / "scratch/alphafold/inputs", arcname="inputs")
    inventory = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "publication_manifest.json":
            with path.open("rb") as stream:
                inventory[str(path.relative_to(stage))] = {
                    "bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    manifest = {"destination": DESTINATION, "files": inventory,
                "raw_archives": "contact_rollouts.tar.gz: original completions, votes, timing, truth; helico_inputs.tar.gz: exact input CIFs, sequences, residue maps and ranked contact pairs; helico_coordinates.tar.gz: original diffusion coordinates, contact states, per-sample scores and timings; structured_esmfold2_predictions.tar.gz: all 500 seeded ESMFold2 structures and timings; structured_helico_coordinates.tar.gz: all 505 contact states, 1515 diffusion coordinates, scores and timings; af2/af3/boltz2_predictions.tar.gz: all candidates, confidence selection and timings; alphafold_inputs.tar.gz: shared MSA queries/alignments, no weights"}
    (stage / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "data/publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(inventory)} files, {sum(r['bytes'] for r in inventory.values()) / 1e6:.1f} MB → {DESTINATION}")
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), DESTINATION], check=True)


if __name__ == "__main__":
    main()
