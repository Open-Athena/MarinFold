"""Publish AF3 sampling outputs and provenance, excluding all private weights."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/af3-sampling-2026-10-06"


def main() -> None:
    """Assemble a public, checksummed package; --upload synchronizes it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    stage = HERE / "scratch/public_af3_sampling"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "data/af3_sampling_protocol.json").read_text())
    with (HERE / "data/af3_sampling_samples.csv").open() as stream:
        references = {row["stem"]: row["ground_truth_file"] for row in csv.DictReader(stream)}
    paths = [HERE / name for name in ["AF3_SAMPLING.md", "prepare_af3_sampling.py", "render_af3_sampling.py",
             "publish_af3_sampling.py", "theme.py", "build_summary.py", "summary_narrative.md",
             "pyproject.toml", "uv.lock", "generation/run_af3_sampling.py", "generation/run_af_baselines.py",
             "generation/alphafold_worker.py", "generation/score_af3_sampling.py", "generation/score_alphafold.py",
             "generation/pyproject.toml", "generation/uv.lock", "plots/summary.pdf", "test_af3_sampling.py"]]
    for pattern in ["data/af3_sampling*", "plots/01b_af3_sampling*", "site/01b_af3_sampling*",
                    "data/inputs/Lato-*.ttf", "data/af3_*terms*", "data/af3_notice.txt"]:
        paths.extend(p for p in HERE.glob(pattern) if p.name != "af3_sampling_publication.json")
    for path in sorted(set(paths)):
        destination = stage / path.relative_to(HERE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    raw = stage / "raw"
    raw.mkdir(exist_ok=True)
    for record in protocol["targets"]:
        stem = record["stem"]
        shutil.copyfile(HERE / "scratch/af3_sampling" / f"{stem}.tar.gz", raw / f"{stem}.tar.gz")
        for source, folder in [
            (HERE / "scratch/alphafold/inputs" / f"{stem}.a3m.gz", "msa"),
            (Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / references[stem], "ground_truth"),
        ]:
            destination = raw / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    with tarfile.open(raw / "original_af3_5x5.tar.gz", "w:gz") as archive:
        for record in protocol["targets"]:
            stem = record["stem"]
            source = HERE / "scratch/alphafold/results/af3" / stem
            archive.add(source, arcname=stem)
    inventory = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "publication_manifest.json":
            with path.open("rb") as stream:
                inventory[str(path.relative_to(stage))] = dict(bytes=path.stat().st_size,
                                                             sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    manifest = dict(destination=DESTINATION, files=inventory,
                    extraction="Extract raw/<stem>.tar.gz into scratch/af3_sampling/results/; "
                    "extract original_af3_5x5.tar.gz into scratch/alphafold/results/af3/. "
                    "Per-sample CSVs contain relative paths and checksums. MSA and reference structures are under raw/.")
    (stage / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "data/af3_sampling_publication.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(inventory)} artifacts, {sum(r['bytes'] for r in inventory.values()) / 1e6:.1f} MB → {DESTINATION}")
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), DESTINATION, "--quiet"], check=True)


if __name__ == "__main__":
    main()
