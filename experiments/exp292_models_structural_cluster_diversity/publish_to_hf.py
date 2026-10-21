"""Publish a reproducible small curation bundle to the public MarinFold bucket."""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from structure_audit import read_csv


def main() -> None:
    """Rebuild the gallery and publish measurements plus its validated C-alpha inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--use-existing-gallery", action="store_true")
    args = parser.parse_args()
    if not args.name.replace("-", "").isalnum():
        raise ValueError("Artifact name must contain only letters, digits and hyphens")
    root = Path(__file__).resolve().parent
    if not args.use_existing_gallery:
        subprocess.run(
            [
                sys.executable,
                str(root / "build_gallery.py"),
                "--sample-dir",
                str(args.sample_dir),
                "--cache",
                str(args.cache),
                "--output",
                str(args.gallery),
            ],
            check=True,
        )
    destination = "hf://buckets/open-athena/MarinFold/data/exp292/" + args.name
    source = json.loads((args.sample_dir / "audit.json").read_text())["source"]
    attribution = {
        "afdb": "Derived from AlphaFold Protein Structure Database v4 (Google DeepMind and EMBL-EBI), CC BY 4.0: https://alphafold.ebi.ac.uk/faq .",
        "esmfold2": "Derived from Biohub ESM Atlas v1 ESMFold2 predictions, accessed 2026-09-14, CC BY-SA 4.0: https://registry.opendata.aws/biohub-esm-atlas/ . Adapted data are shared under CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ .",
    }[source]
    with tempfile.TemporaryDirectory(prefix="exp292-publish-") as directory:
        stage = Path(directory)
        shutil.copytree(args.sample_dir, stage / "measurements")
        shutil.copy2(args.gallery, stage / "index.html")
        shutil.copy2(
            args.gallery.with_name("3Dmol-LICENSE.txt"), stage / "3Dmol-LICENSE.txt"
        )
        with tarfile.open(stage / "ca-structures.tar.gz", "w:gz") as archive:
            for row in read_csv(args.sample_dir / "sample.csv"):
                path = args.cache / f"{row['entry_id']}.npz"
                archive.add(path, arcname=path.name)
        (stage / "README.md").write_text(
            "# Exp292 structural diversity curation\n\n"
            "Developmental inspection data for https://github.com/Open-Athena/MarinFold/issues/292. "
            "No candidate is cleared for training. The small stratified sample is not a population yield estimate.\n\n"
            "Download index.html and open it in a browser for the offline interactive gallery. "
            "ca-structures.tar.gz contains source C-alpha arrays, sequences and residue confidence used by build_gallery.py. "
            "The measurements include original source IDs, hashes, provenance and pair metrics. "
            "These are original AFDB v4 or ESMFold2 Atlas predictions, not MarinFold predictions. "
            "Source attributions and scientific caveats are in the experiment README.\n\n"
            + attribution
            + "\nCoordinates were reduced to C-alpha traces for inspection; source sequence and confidence were retained. The included 3Dmol runtime has its separate BSD license.\n"
        )
        subprocess.run(["hf", "buckets", "sync", str(stage), destination], check=True)
    print(
        json.dumps(
            {"published": destination, "gallery": destination + "/index.html"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
