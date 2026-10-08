"""Publish the sparse-oracle experiment and its exact conditioning/coordinates."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis"


def digest(path: Path) -> str:
    """Hash an artifact without keeping its bytes in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Build a reproducible public bundle after the complete sweep is validated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="Recover raw coordinates from the durable Modal volume")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--l2", action="store_true", help="Publish the 305-protein L/2 follow-up")
    parser.add_argument("--seed-completion", action="store_true", help="Publish the paired contact-completion follow-up")
    args = parser.parse_args()
    if args.l2 and args.seed_completion:
        raise ValueError("Choose one publication phase")
    phase = "seed_completion" if args.seed_completion else ("oracle_l2" if args.l2 else "oracle_budget")
    destination = PUBLIC_ROOT + "/" + phase.replace("_", "-") + "-2026-10-08"
    manifest = json.loads((HERE / f"data/{phase}_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(HERE / "data" / name) != expected:
            raise ValueError(f"Stale oracle-budget result: {name}")
    if args.archive:
        subprocess.run(["uv", "run", "--project", "generation", "modal", "run",
                        "generation/archive_helico.py", "--" + phase.replace("_", "-")], cwd=HERE, check=True)
    stage = HERE / "scratch" / f"public_{phase}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for pattern in ("data/*.csv", "data/*.json", "data/*.txt", "data/*.md", "data/inputs/Lato-*", "plots/*", "site/*",
                    "*oracle_budget*.py", "*oracle_l2*.py", "*seed_completion*.py", "generation/*seed_completion*.py",
                    "generation/run_contacts.py", "generation/score_rollout_worker.py", "generation/checkpoint_manifest.json",
                    "generation/*oracle_budget*.py", "generation/run_helico.py",
                    "generation/archive_helico.py", "generation/pyproject.toml", "generation/uv.lock",
                    "theme.py", "poster_style.py", "prepare.py", "render*.py", "plan_missing.py",
                    "build_summary.py", "sources_remote.json", "pyproject.toml", "uv.lock", "*.md"):
        for source in HERE.glob(pattern):
            if not source.is_file() or source.name == f"{phase}_publication.json":
                continue
            target = stage / source.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    with tarfile.open(stage / f"{phase}_inputs.tar.gz", "w:gz") as archive:
        archive.add(HERE / "scratch/helico" / (phase if phase != "oracle_budget" else "oracle_budget_low_msa"), arcname=f"{phase}_inputs")
        if args.seed_completion:
            archive.add(HERE / "scratch/seed_completion/inputs", arcname="marinfold_seed_prompts")
    shutil.copyfile(HERE / f"scratch/{phase}_coordinates.tar.gz", stage / f"{phase}_coordinates.tar.gz")
    if args.seed_completion:
        shutil.copyfile(HERE / "scratch/seed_completion/rollouts.tar.gz", stage / "seed_completion_rollouts.tar.gz")
        shutil.copyfile(HERE / "scratch/oracle_budget_coordinates.tar.gz", stage / "reused_direct_coordinates.tar.gz")
    inventory = {str(path.relative_to(stage)): dict(bytes=path.stat().st_size, sha256=digest(path))
                 for path in sorted(stage.rglob("*")) if path.is_file()}
    publication = dict(destination=destination, files=inventory,
        inputs="Every ground-truth CIF, frozen sequence, sampled conditioning matrix, random seed and map hash",
        coordinates="Every diffusion structure, exact contact_state, per-sample metric/confidence and timing")
    (stage / "publication_manifest.json").write_text(json.dumps(publication, indent=2) + "\n")
    (HERE / f"data/{phase}_publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(f"{len(inventory)} files, {sum(row['bytes'] for row in inventory.values()) / 1e6:.1f} MB → {destination}")
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), destination, "--quiet"], check=True)


if __name__ == "__main__":
    main()
