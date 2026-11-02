"""Rehydrate exact ESM source backbones from the published compact artifacts.

The original source-CIF extraction has already been materialized into the
project's public data bucket. A rebuild fetches only the selected compact PDBs
and verifies their committed SHA-256 digests, avoiding repeated multi-GB
transfers of public source Parquet row groups.
"""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import gemmi


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STRUCTURES = HERE / "structures"
MANIFEST = DATA / "structure_manifest.json"
BUCKET_PREFIX = "data/training-explorer/2026-09-22/structures"
RESOLVE = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
AA3 = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        strict=True,
    )
)
BACKBONE = {"N", "CA", "C", "O"}


def backbone_pdb(cif: str, sequence: str) -> str:
    """Keep one source backbone and label residues with the document sequence."""
    structure = gemmi.read_structure_string(cif, format=gemmi.CoorFormat.Mmcif)
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    if len(structure) != 1:
        raise ValueError("Expected one model in the source prediction")
    residues = [residue for chain in structure[0] for residue in chain]
    if len(residues) != len(sequence):
        raise ValueError(
            f"Source structure has {len(residues)} residues; document has {len(sequence)}"
        )
    for residue, letter in zip(residues, sequence, strict=True):
        if letter not in AA3:
            raise ValueError(f"Noncanonical residue {letter} cannot be relabeled")
        residue.name = AA3[letter]
        for index in range(len(residue) - 1, -1, -1):
            if residue[index].name not in BACKBONE:
                del residue[index]
    return "".join(
        f"{line.rstrip()}\n" for line in structure.make_pdb_string().splitlines()
    )


def restore(protein: dict, expected: dict) -> str:
    """Fetch one compact PDB if needed and verify bytes and sequence labels."""
    filename = hashlib.sha256(protein["id"].encode()).hexdigest()[:20] + ".pdb"
    if expected["file"] != filename:
        raise ValueError(f"Manifest filename mismatch for {protein['id']}")
    path = STRUCTURES / filename
    if path.exists():
        content = path.read_bytes()
    else:
        url = RESOLVE + quote(f"{BUCKET_PREFIX}/{filename}", safe="")
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
    if (
        len(content) != expected["bytes"]
        or hashlib.sha256(content).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"Published structure checksum mismatch for {protein['id']}")
    if not path.exists():
        path.write_bytes(content)
    structure = gemmi.read_structure(str(path))
    residues = [residue for chain in structure[0] for residue in chain]
    if len(residues) != len(protein["sequence"]):
        raise ValueError(f"Backbone length mismatch for {protein['id']}")
    for residue, letter in zip(residues, protein["sequence"], strict=True):
        if residue.name != AA3[letter] or not any(
            atom.name == "CA" for atom in residue
        ):
            raise ValueError(f"Backbone sequence mismatch for {protein['id']}")
    return filename


def main() -> None:
    """Restore the 179 ESM previews for this fixed, uniform sample."""
    snapshots = {
        name: json.loads((DATA / f"{name}.json").read_text())
        for name in ("latest", "original")
    }
    manifest = {item["proteinId"]: item for item in json.loads(MANIFEST.read_text())}
    esm = [
        protein
        for snapshot in snapshots.values()
        for protein in snapshot["proteins"]
        if "ESM-Atlas" in protein["source"]
    ]
    if len(esm) != 179:
        raise ValueError(
            f"Expected 179 ESM previews for the pinned sample, got {len(esm)}"
        )
    STRUCTURES.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        filenames = list(
            pool.map(lambda protein: restore(protein, manifest[protein["id"]]), esm)
        )
    for protein, filename in zip(esm, filenames, strict=True):
        protein["structureUrl"] = f"structures/{filename}"
        protein["structureFormat"] = "pdb"
        protein["structureNote"] = (
            "Source backbone · MPNN sequence"
            if protein["source"].startswith("MPNN")
            else "Exact ESMFold2 source backbone"
        )
    for name, snapshot in snapshots.items():
        (DATA / f"{name}.json").write_text(json.dumps(snapshot, separators=(",", ":")))
    print(f"Restored and verified {len(esm)} compact ESM backbones", flush=True)


if __name__ == "__main__":
    main()
