"""Recover all new Helico coordinates and control maps from the durable volume."""

import io
import tarfile
from pathlib import Path

import modal

app = modal.App("marinfold-exp325-archive")
volume = modal.Volume.from_name("marinfold-exp325-helico-results")


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={"/results": volume}, region="us-east", cpu=4, timeout=1200)
def archive() -> bytes:
    """Package only this experiment's two completed output prefixes."""
    volume.reload()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for phase in ("confidence", "folding"):
            root = Path("/results") / f"exp325-exp277-step266344-{phase}-v1"
            if not root.is_dir():
                raise FileNotFoundError(root)
            output.add(root, arcname=phase)
    return buffer.getvalue()


@app.local_entrypoint()
def run() -> None:
    """Save the public raw-artifact payload without loading a predictor."""
    destination = Path(__file__).resolve().parent.parent / "scratch/helico_coordinates.tar.gz"
    destination.write_bytes(archive.remote())
    print(f"Recovered {destination.stat().st_size / 1e6:.1f} MB to {destination}")
