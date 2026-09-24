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


def main() -> None:
    """Build a checksummed public package from frozen local results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    stage = HERE / "scratch/public"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for folder in ("data", "plots", "site"):
        shutil.copytree(HERE / folder, stage / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("publication_manifest.json"))
    for name in ("README.md", "DRAFT.md", "FIGURES.md", "RUNBOOK.md", "summary_narrative.md",
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
