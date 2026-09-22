"""Materialize compact, exact ESM source backbones for sampled proteins.

Each ESM source Parquet is accessed by HTTP range. Only the row group
containing a sampled protein is read; the script enforces an 8 GB aggregate
compressed-byte budget, below the repo's 10 GB cross-region sign-off limit.
"""

import gzip
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import fsspec
import gemmi
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STRUCTURES = HERE / "structures"
MAP = (
    HERE.parents[1]
    / "experiments/exp266_data_mpnn_redesign_contacts_v1/esm_shard_map.json.gz"
)
SOURCE_URL = (
    "https://huggingface.co/buckets/open-athena/esm-atlas-esmfold2-distill/"
    "resolve/structures%2Fparts%2Fpart_{part:05d}.parquet"
)
MAX_COMPRESSED_BYTES = 8_000_000_000
AA3 = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        strict=True,
    )
)
BACKBONE = {"N", "CA", "C", "O"}


def candidate_parts(protein: dict, reverse: dict[int, list[int]]) -> list[int]:
    """Find the source part(s) allowed by the corpus' recorded shard map."""
    if protein["id"].startswith("original:"):
        return [int(protein["targetId"].split("|", 1)[1].split("_", 1)[0])]
    filename = protein["sourceFile"].rsplit("/", 1)[-1]
    if protein["source"].startswith("MPNN"):
        match = re.match(r"documents-(\d+)-of-", filename)
        if match is None:
            raise ValueError(f"Unrecognized MPNN ESM source file: {filename}")
        return [int(match.group(1))]
    match = re.match(r"shard-(\d+)-of-", filename)
    if match is None:
        raise ValueError(f"Unrecognized native ESM source file: {filename}")
    result = reverse.get(int(match.group(1)), [])
    if not result:
        raise ValueError(f"No source part for decontaminated shard {filename}")
    return result


@lru_cache(maxsize=512)
def source_metadata(part: int) -> pq.FileMetaData:
    """Read the footer of one source part, with no CIF transfer."""
    with fsspec.open(SOURCE_URL.format(part=part), "rb", block_size=1 << 20) as stream:
        return pq.read_metadata(stream)


def group_in_part(part: int, target: str) -> tuple[int, int, int] | None:
    """Find a target's sorted row group within one candidate part."""
    metadata = source_metadata(part)
    first = metadata.row_group(0).column(0).statistics.min
    last = metadata.row_group(metadata.num_row_groups - 1).column(0).statistics.max
    if not first <= target <= last:
        return None
    candidates = [
        group
        for group in range(metadata.num_row_groups)
        if (stats := metadata.row_group(group).column(0).statistics)
        and stats.min <= target <= stats.max
    ]
    if not candidates:
        return None
    # Groups overlap in their min/max ranges even though source parts are
    # ordered. Check the small entry_id column before fetching bulky CIFs.
    with fsspec.open(SOURCE_URL.format(part=part), "rb", block_size=1 << 20) as stream:
        parquet = pq.ParquetFile(stream)
        for group in candidates:
            ids = (
                parquet.read_row_group(group, columns=["entry_id"])
                .column(0)
                .to_pylist()
            )
            if target in ids:
                compressed = sum(
                    metadata.row_group(group).column(index).total_compressed_size
                    for index in (0, 1)
                )
                return part, group, compressed
    return None


def find_source(protein: dict, reverse: dict[int, list[int]]) -> tuple[int, int, int]:
    """Use source-part and row-group ranges to locate the exact CIF."""
    target = protein["entryId"]
    for part in candidate_parts(protein, reverse):
        found = group_in_part(part, target)
        if found is not None:
            return found
    # The last original-corpus shards were repacked after their source parts.
    # The source parts themselves are globally ordered by entry ID, so a
    # footer-only binary search locates those records without a bulk scan.
    low, high = 0, 3337
    while low <= high:
        part = (low + high) // 2
        metadata = source_metadata(part)
        first = metadata.row_group(0).column(0).statistics.min
        last = metadata.row_group(metadata.num_row_groups - 1).column(0).statistics.max
        if target < first:
            high = part - 1
        elif target > last:
            low = part + 1
        else:
            found = group_in_part(part, target)
            if found is not None:
                return found
            break
    raise ValueError(f"No source structure found for {protein['id']} ({target})")


def backbone_pdb(cif: str, sequence: str) -> str:
    """Preserve source coordinates and use the document's residue identities."""
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


def fetch_group(part: int, group: int, proteins: list[dict]) -> list[tuple[str, str]]:
    """Read one selected row group and return PDB text for its requested IDs."""
    url = SOURCE_URL.format(part=part)
    with fsspec.open(url, "rb", block_size=1 << 20) as stream:
        parquet = pq.ParquetFile(stream)
        table = parquet.read_row_group(group, columns=["entry_id", "cif_content"])
    by_entry = {row["entry_id"]: row["cif_content"] for row in table.to_pylist()}
    outputs = []
    for protein in proteins:
        cif = by_entry.get(protein["entryId"])
        if cif is None:
            raise ValueError(f"Row group {part}/{group} lacks {protein['entryId']}")
        outputs.append((protein["id"], backbone_pdb(cif, protein["sequence"])))
    return outputs


def main() -> None:
    """Build all sampled ESM backbone previews and update both JSON catalogs."""
    snapshots = {
        name: json.loads((DATA / f"{name}.json").read_text())
        for name in ("latest", "original")
    }
    proteins = [p for snapshot in snapshots.values() for p in snapshot["proteins"]]
    mapping = json.load(gzip.open(MAP, "rt"))
    reverse: dict[int, list[int]] = defaultdict(list)
    for part, shards in mapping.items():
        for shard in shards:
            reverse[shard].append(int(part))
    esm = [p for p in proteins if "ESM-Atlas" in p["source"]]
    print(f"Locating {len(esm)} ESM source structures", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        locations = list(pool.map(lambda protein: find_source(protein, reverse), esm))
    jobs: dict[tuple[int, int], list[dict]] = defaultdict(list)
    sizes: dict[tuple[int, int], int] = {}
    for protein, (part, group, compressed) in zip(esm, locations, strict=True):
        jobs[(part, group)].append(protein)
        sizes[(part, group)] = compressed
    planned = sum(sizes.values())
    print(
        f"Reading {len(jobs)} source row groups, {planned / 1e9:.2f} GB compressed",
        flush=True,
    )
    if planned > MAX_COMPRESSED_BYTES:
        raise ValueError("Source extraction exceeds the 8 GB transfer budget")
    STRUCTURES.mkdir(exist_ok=True)
    keys = sorted(jobs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda key: fetch_group(*key, jobs[key]), keys)
        for index, structures in enumerate(results, 1):
            for protein_id, pdb in structures:
                filename = hashlib.sha256(protein_id.encode()).hexdigest()[:20] + ".pdb"
                (STRUCTURES / filename).write_text(pdb)
                protein = next(p for p in esm if p["id"] == protein_id)
                protein["structureUrl"] = f"structures/{filename}"
                protein["structureFormat"] = "pdb"
                protein["structureNote"] = (
                    "Source backbone · MPNN sequence"
                    if protein["source"].startswith("MPNN")
                    else "Exact ESMFold2 source backbone"
                )
            if index % 10 == 0 or index == len(keys):
                print(f"Extracted {index}/{len(keys)} row groups", flush=True)
    for name, snapshot in snapshots.items():
        (DATA / f"{name}.json").write_text(json.dumps(snapshot, separators=(",", ":")))
    print(f"Wrote {len(esm)} compact PDB files", flush=True)


if __name__ == "__main__":
    main()
