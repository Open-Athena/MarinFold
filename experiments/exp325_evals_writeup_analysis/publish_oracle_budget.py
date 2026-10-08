"""Publish the sparse-oracle experiment and its exact conditioning/coordinates."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/oracle-budget-2026-10-08"


def digest(path: Path) -> str:
    """Hash an artifact without keeping its bytes in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Build a reproducible public bundle after the complete sweep is validated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="Recover raw coordinates from the durable Modal volume")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "data/oracle_budget_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(HERE / "data" / name) != expected:
            raise ValueError(f"Stale oracle-budget result: {name}")
    if args.archive:
        subprocess.run(["uv", "run", "--project", "generation", "modal", "run",
                        "generation/archive_helico.py", "--oracle-budget"], cwd=HERE, check=True)
    stage = HERE / "scratch/public_oracle_budget"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for pattern in ("data/oracle_budget*", "data/helico_oracle_budget*", "data/figure_rows.csv",
                    "data/inputs/Lato-*", "plots/02e*", "plots/02f*", "plots/summary.pdf", "site/*",
                    "*oracle_budget*.py", "generation/*oracle_budget*.py", "generation/run_helico.py",
                    "generation/archive_helico.py", "generation/pyproject.toml", "generation/uv.lock",
                    "theme.py", "poster_style.py", "prepare.py", "render.py", "plan_missing.py",
                    "build_summary.py", "pyproject.toml", "uv.lock", "README.md", "DRAFT.md",
                    "ORACLE_BUDGET.md", "POSTER.md", "FIGURES.md", "summary_narrative.md"):
        for source in HERE.glob(pattern):
            if not source.is_file():
                continue
            target = stage / source.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    with tarfile.open(stage / "oracle_budget_inputs.tar.gz", "w:gz") as archive:
        archive.add(HERE / "scratch/helico/oracle_budget", arcname="oracle_budget_inputs")
    shutil.copyfile(HERE / "scratch/oracle_budget_coordinates.tar.gz", stage / "oracle_budget_coordinates.tar.gz")
    inventory = {str(path.relative_to(stage)): dict(bytes=path.stat().st_size, sha256=digest(path))
                 for path in sorted(stage.rglob("*")) if path.is_file()}
    publication = dict(destination=DESTINATION, files=inventory,
        inputs="Every ground-truth CIF, frozen sequence, full oracle and sampled conditioning matrix",
        coordinates="Every diffusion structure, exact contact_state, per-sample metric/confidence and timing")
    (stage / "publication_manifest.json").write_text(json.dumps(publication, indent=2) + "\n")
    (HERE / "data/oracle_budget_publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(f"{len(inventory)} files, {sum(row['bytes'] for row in inventory.values()) / 1e6:.1f} MB → {DESTINATION}")
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), DESTINATION, "--quiet"], check=True)


if __name__ == "__main__":
    main()
