"""Recover all new Helico coordinates and control maps from the durable volume."""

import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
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


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={"/results": volume}, region="us-east", cpu=4, timeout=1200)
def archive_structured() -> bytes:
    """Keep every structured-decoy map and every downstream diffusion coordinate."""
    volume.reload()
    root = Path("/results/exp325-exp277-step266344-structured-v1")
    stems = {"8ii8_A", "8oxk_A", "8qoh_A", "8ux2_A", "8wrx_A"}
    if {p.name for p in root.iterdir()} != stems:
        raise ValueError("Unexpected structured-decoy targets")
    names = {f"{key}.{suffix}" for key in ["oracle-0", *(f"esmfold2-{i}" for i in range(100))]
             for suffix in ("json", "npz")}
    paths = []
    for stem in sorted(stems):
        if {p.name for p in (root / stem).iterdir()} != names:
            raise ValueError(f"Incomplete structured-decoy outputs: {stem}")
        paths.extend(root / stem / name for name in sorted(names))
    buffer = io.BytesIO()
    with ThreadPoolExecutor(max_workers=24) as pool, tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for path, content in zip(paths, pool.map(Path.read_bytes, paths), strict=True):
            info = tarfile.TarInfo(str(Path("structured") / path.relative_to(root)))
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={"/results": volume}, region="us-east", cpu=4, memory=4096, timeout=1800)
def archive_oracle_budget(stems: list[str], map_names: list[str]) -> bytes:
    """Archive every frozen budget map and all three diffusion structures."""
    volume.reload()
    root = Path("/results/exp325-exp277-step266344-oracle_budget-v1")
    if {path.name for path in root.iterdir()} != set(stems):
        raise ValueError("Oracle-budget target inventory differs")
    expected = {f"{key}.{extension}" for key in map_names for extension in ("json", "npz")}
    paths = []
    for stem in sorted(stems):
        if {path.name for path in (root / stem).iterdir()} != expected:
            raise ValueError(f"Incomplete oracle-budget outputs: {stem}")
        paths.extend(root / stem / name for name in sorted(expected))
    buffer = io.BytesIO()
    with ThreadPoolExecutor(max_workers=24) as pool, tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in zip(paths, pool.map(Path.read_bytes, paths), strict=True):
            info = tarfile.TarInfo(str(Path("oracle_budget") / path.relative_to(root)))
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@app.local_entrypoint()
def run(structured: bool = False, oracle_budget: bool = False) -> None:
    """Save the public raw-artifact payload without loading a predictor."""
    if structured and oracle_budget:
        raise ValueError("Choose exactly one archive phase")
    if oracle_budget:
        root = Path(__file__).resolve().parent.parent
        protocol = json.loads((root / "data/oracle_budget_protocol.json").read_text())
        destination = root / "scratch/oracle_budget_coordinates.tar.gz"
        destination.write_bytes(archive_oracle_budget.remote(list(protocol["files_sha256"]), protocol["map_keys"]))
        print(f"Recovered {destination.stat().st_size / 1e6:.1f} MB to {destination}")
        return
    filename = "structured_decoys/helico_coordinates.tar.gz" if structured else "helico_coordinates.tar.gz"
    destination = Path(__file__).resolve().parent.parent / "scratch" / filename
    destination.write_bytes(archive_structured.remote() if structured else archive.remote())
    print(f"Recovered {destination.stat().st_size / 1e6:.1f} MB to {destination}")
