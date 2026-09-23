"""Stage fixed inputs and private, already-available weights into our Modal volume."""

import json
from pathlib import Path

import modal

from prepare_alphafold import digest

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Validate local weights and upload once; never publish these parameters."""
    manifest = json.loads((ROOT / "data/alphafold_inputs.json").read_text())
    volume = modal.Volume.from_name("marinfold-exp325-alphafold", create_if_missing=True)
    with volume.batch_upload(force=True) as upload:
        upload.put_directory(ROOT / "scratch/alphafold/inputs", "/inputs")
        for name, pin in manifest["weights"].items():
            path = (Path.home() / ".cache/colabfold/params" / Path(name).name
                    if name.startswith("af2/") else Path.home() / "Dropbox/AF3/params-20241113/af3.bin.zst")
            if digest(path) != pin["sha256"]:
                raise ValueError(f"Weights changed: {name}")
            upload.put_file(path, "/weights/" + name)
    # Save the final protocol after the larger immutable files have transferred.
    with volume.batch_upload(force=True) as upload:
        upload.put_file(ROOT / "data/alphafold_inputs.json", "/protocol.json")
    print("Fixed inputs and private weights staged", flush=True)


if __name__ == "__main__":
    main()
