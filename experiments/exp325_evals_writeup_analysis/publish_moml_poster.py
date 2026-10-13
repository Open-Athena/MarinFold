"""Publish the reviewed MoML poster and its numerical provenance.

Run revise_moml_poster.py first. This copies only the final reviewed outputs,
panels and provenance; it does not publish the original attachment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/moml-poster-2026-10-13"
PUBLIC = DESTINATION.replace("hf://buckets/", "https://huggingface.co/buckets/").replace(
    "/MarinFold/data/", "/MarinFold/resolve/data/")


def sha256(path: Path) -> str:
    """Return the exact artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Assemble a hash-checked publication bundle and optionally upload it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "data/moml_poster_manifest.json").read_text())
    results = HERE / "scratch/moml_poster/revised"
    stage = HERE / "scratch/moml_poster/public"
    stage.mkdir(parents=True, exist_ok=True)
    for filename, digest in manifest["outputs"].items():
        source = results / filename
        if sha256(source) != digest:
            raise ValueError(f"Changed poster output: {filename}; rebuild before publishing")
        shutil.copyfile(source, stage / filename)
    for filename in ["MOML_POSTER_REVIEW.md", "revise_moml_poster.py", "publish_moml_poster.py",
                     "theme.py", "data/moml_poster_manifest.json", "data/moml_poster_figure_rows.csv"]:
        target = stage / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / filename, target)
    for filename in manifest["inputs"]:
        source = HERE.parent.parent / filename
        if sha256(source) != manifest["inputs"][filename]:
            raise ValueError(f"Changed analysis input: {filename}")
        target = stage / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copytree(HERE / "plots/moml", stage / "panels", dirs_exist_ok=True)
    publication = {"destination": DESTINATION,
                   "pptx": PUBLIC + "/MarinFold_MoML_revised.pptx",
                   "pdf": PUBLIC + "/MarinFold_MoML_revised.pdf",
                   "png": PUBLIC + "/MarinFold_MoML_revised.png",
                   "files": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), DESTINATION, "--quiet"], check=True)
        (HERE / "data/moml_poster_publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(json.dumps({k: v for k, v in publication.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
