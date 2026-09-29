"""Materialize exact staged MPNN AFDB backbones and native AFDB previews."""

import configparser
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import gemmi
import pyarrow.parquet as pq
import s3fs

from extract_structures import AA3, DATA, STRUCTURES, backbone_pdb


BUCKET = "marin-us-east-02a"


def staged_backbone_pdb(row: dict, sequence: str) -> str:
    """Rebuild the exact #266 source backbone with the sampled design sequence."""
    coords = row["coords_milli"]
    plddt = row["ca_plddt"]
    if len(sequence) != len(row["sequence"]) or len(coords) != len(sequence) * 12:
        raise ValueError(f"Staged backbone length mismatch for {row['entry_id']}")
    structure = gemmi.Structure()
    structure.name = row["entry_id"]
    model = gemmi.Model("1")
    chain = gemmi.Chain(row["chain_id"])
    for index, letter in enumerate(sequence):
        residue = gemmi.Residue()
        residue.name = AA3[letter]
        residue.seqid = gemmi.SeqId(int(row["resnum_start"]) + index, " ")
        residue.het_flag = "A"
        for atom_index, name in enumerate(("N", "CA", "C", "O")):
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(name[0])
            base = index * 12 + atom_index * 3
            atom.pos = gemmi.Position(
                *(coords[base + axis] / 1000 for axis in range(3))
            )
            atom.b_iso = plddt[index]
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    return "".join(
        f"{line.rstrip()}\n" for line in structure.make_pdb_string().splitlines()
    )


def staged_row(fs: s3fs.S3FileSystem, protein: dict) -> dict:
    """Use the 8-documents-per-parent ordering to read one selected row group."""
    filename = protein["sourceFile"].rsplit("/", 1)[-1]
    match = re.match(r"documents-backbones-(\d+)-of-00199.parquet", filename)
    if match is None:
        raise ValueError(f"Unexpected MPNN AFDB file: {filename}")
    part = int(match.group(1))
    parent_index = int(protein["sourceRow"]) // 8
    path = f"{BUCKET}/MarinFold/exp266/backbones/backbones-{part:05d}-of-00199.parquet"
    with fs.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        start = 0
        for group in range(parquet.metadata.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if start <= parent_index < start + size:
                row = (
                    parquet.read_row_group(group)
                    .slice(parent_index - start, 1)
                    .to_pylist()[0]
                )
                if row["entry_id"] != protein["entryId"]:
                    raise ValueError(
                        f"Parent entry mismatch: {row['entry_id']} != {protein['entryId']}"
                    )
                return row
            start += size
    raise ValueError(f"No staged parent at row {parent_index}")


def current_afdb_cif(entry: str) -> str:
    """Fetch the current public AlphaFold DB model for a native accession."""
    accession = entry.split("-")[1]
    request = urllib.request.Request(
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
        headers={"User-Agent": "MarinFold-training-explorer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        predictions = json.load(response)
    match = next(
        (
            prediction
            for prediction in predictions
            if prediction["modelEntityId"] == entry
        ),
        None,
    )
    if match is None:
        raise ValueError(f"No AlphaFold DB model for {entry}")
    with urllib.request.urlopen(match["cifUrl"], timeout=60) as response:
        return response.read().decode()


def main() -> None:
    """Add 3D coordinates for every AFDB-sourced sample."""
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    fs = s3fs.S3FileSystem(
        key=cw["aws_access_key_id"],
        secret=cw["aws_secret_access_key"],
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    STRUCTURES.mkdir(exist_ok=True)
    for name in ("latest", "original"):
        path = DATA / f"{name}.json"
        snapshot = json.loads(path.read_text())
        for protein in snapshot["proteins"]:
            if "AFDB" not in protein["source"]:
                continue
            if protein["source"] == "MPNN AFDB":
                pdb = staged_backbone_pdb(staged_row(fs, protein), protein["sequence"])
                note = "Exact source backbone · MPNN sequence"
            else:
                pdb = backbone_pdb(
                    current_afdb_cif(protein["entryId"]), protein["sequence"]
                )
                note = "Current AFDB model for source accession"
            filename = hashlib.sha256(protein["id"].encode()).hexdigest()[:20] + ".pdb"
            (STRUCTURES / filename).write_text(pdb)
            protein["structureUrl"] = f"structures/{filename}"
            protein["structureFormat"] = "pdb"
            protein["structureNote"] = note
            print(f"{name}: {protein['label']}", flush=True)
        path.write_text(json.dumps(snapshot, separators=(",", ":")))


if __name__ == "__main__":
    main()
