# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 -- cheap mmCIF *header* scan of the whole local PDB mirror.

One parquet row per PDB entry with everything the entry-level curation
filters need (release date, resolution, experimental method, polymer entity
types and lengths, assembly-1 composition), read straight out of the mmCIF
header categories. No coordinates are touched, so this is ~40 ms/entry
rather than the seconds a pyconfind run costs -- worth doing first so the
expensive pass in ``curate_and_generate.py`` only visits entries that can
possibly survive.

Everything here is *entry* level. Chain-level quality filters (all-unknown
chains, unresolved chains, CA breaks, inter-chain clashes) need coordinates
and are applied in the later pass.

Usage::

    uv run python scan_metadata.py --mirror /data/tim/af3-db/mmcif_files \
        --out /data/exp222_pdb_curation/metadata/entries.parquet
"""

import argparse
import os
import sys
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import gemmi
import pyarrow as pa
import pyarrow.parquet as pq


# Polymer types (``_entity_poly.type``) we treat as protein. The PDB
# vocabulary distinguishes L- and D-peptide chains and the "linking"
# variants; contacts-v1's residue vocabulary is the canonical 20 L-amino
# acids, so only the L-polypeptide entities are candidates. Everything else
# (nucleic acids, sugars, D-peptides, peptide nucleic acids) is a
# non-protein entity for our purposes and is ignored per the issue.
PROTEIN_ENTITY_TYPES = frozenset({"polypeptide(l)"})

# Order matters: the first category that carries a usable number wins.
# X-ray/neutron refinement resolution, then the cryo-EM reconstruction
# resolution, then the raw reflection statistics as a last resort.
_RESOLUTION_SOURCES = (
    ("_refine.", "ls_d_res_high"),
    ("_em_3d_reconstruction.", "resolution"),
    ("_reflns.", "d_resolution_high"),
)


def _first(values: list[Any] | None) -> str | None:
    """First non-null, non-``?``/``.`` value of an mmCIF column."""
    if not values:
        return None
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s not in ("?", "."):
            return s
    return None


def _release_date(block: gemmi.cif.Block) -> str | None:
    """Initial release date = the ``ordinal == 1`` revision-history row.

    This is the date Protenix/AF3 cut on, and it is *not* the deposition
    date (``_pdbx_database_status.recvd_initial_deposition_date``), which
    can precede release by years. Falls back to the earliest revision date
    if the ordinal column is missing or malformed.
    """
    hist = block.get_mmcif_category("_pdbx_audit_revision_history.")
    if not hist:
        return None
    dates = hist.get("revision_date") or []
    ordinals = hist.get("ordinal") or []
    candidates = [str(d).strip() for d in dates if d and str(d).strip() not in ("?", ".")]
    if not candidates:
        return None
    for ordinal, date in zip(ordinals, dates):
        if ordinal is not None and str(ordinal).strip() == "1" and date:
            return str(date).strip()
    return min(candidates)


def _resolution(block: gemmi.cif.Block) -> tuple[float | None, str | None]:
    """Best available reported resolution in Angstrom, and where it came from."""
    for category, tag in _RESOLUTION_SOURCES:
        raw = _first(block.get_mmcif_category(category).get(tag))
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        return value, category.strip("_.")
    return None, None


def _count_operators(expression: str) -> int:
    """Number of symmetry operators an ``oper_expression`` expands to.

    Handles the three forms in the PDB: a plain list (``1,2,3``), a range
    (``1-60``), and a product of parenthesised groups (``(1-60)(61-88)``).
    Only the count matters here, not the operators themselves.
    """
    expression = expression.strip()
    if not expression:
        return 1
    if "(" in expression:
        total = 1
        depth = 0
        current = ""
        for ch in expression:
            if ch == "(":
                depth += 1
                if depth == 1:
                    current = ""
                    continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    total *= _count_operators(current)
                    continue
            if depth >= 1:
                current += ch
        return total
    count = 0
    for part in expression.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, _, hi = part.partition("-")
            try:
                count += int(hi) - int(lo) + 1
                continue
            except ValueError:
                pass
        count += 1
    return max(count, 1)


def _assembly_protein_chains(
    block: gemmi.cif.Block, assembly_id: str, protein_asyms: set[str]
) -> tuple[int, int]:
    """Protein chain counts for ``assembly_id``: (distinct asyms, built copies).

    An assembly is built by applying one or more symmetry operators to lists
    of ``label_asym_id``s; an asym listed under an expression of *n*
    operators contributes *n* chains to the built assembly. Both numbers are
    counted over **protein** asyms only -- the ``asym_id_list`` also names
    ligand, water and nucleic-acid asyms, which this experiment ignores, so
    counting the raw list length would badly overestimate the complex size
    (a monomer crystallised with three waters would look like a tetramer).

    The distinct count is what the ASU-level bookkeeping needs; the built
    count is the number of ``<n-term>``/``<c-term>`` pairs a multimer
    document would carry. Both are cheap estimates for pre-filtering -- the
    exact expansion needs gemmi's assembly machinery and coordinates, and
    happens in the later pass.
    """
    gen = block.get_mmcif_category("_pdbx_struct_assembly_gen.")
    if not gen:
        return 0, 0
    ids = gen.get("assembly_id") or []
    asym_lists = gen.get("asym_id_list") or []
    opers = gen.get("oper_expression") or []
    distinct: set[str] = set()
    built = 0
    for i, aid in enumerate(ids):
        if aid is None or str(aid).strip() != assembly_id:
            continue
        raw = asym_lists[i] if i < len(asym_lists) else None
        if not raw:
            continue
        chunk = [a.strip() for a in str(raw).split(",") if a.strip()]
        protein_chunk = [a for a in chunk if a in protein_asyms]
        if not protein_chunk:
            continue
        distinct.update(protein_chunk)
        oper = str(opers[i]).strip() if i < len(opers) and opers[i] else "1"
        built += len(protein_chunk) * _count_operators(oper)
    return len(distinct), built


def scan_one(path: str) -> dict[str, Any]:
    """Header-scan a single mmCIF file into a flat metadata row.

    Parse failures are captured in the ``error`` column rather than raised:
    a single malformed entry out of ~196k must not kill the scan, and the
    ledger in ``curate.py`` counts them explicitly so nothing is dropped
    silently.
    """
    pdb_id = Path(path).name.split(".")[0].lower()
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "path": path,
        "release_date": None,
        "deposition_date": None,
        "method": None,
        "resolution": None,
        "resolution_source": None,
        "n_protein_entities": 0,
        "n_protein_asym_chains": 0,
        "protein_entity_ids": [],
        "protein_entity_lengths": [],
        "protein_strand_ids": [],
        "n_assembly1_protein_asyms": 0,
        "n_assembly1_protein_chains": 0,
        "error": None,
    }
    try:
        block = gemmi.cif.read(path).sole_block()
    except Exception as exc:  # noqa: BLE001 - recorded in the row, not swallowed
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["release_date"] = _release_date(block)
    row["deposition_date"] = _first(
        block.get_mmcif_category("_pdbx_database_status.").get(
            "recvd_initial_deposition_date"
        )
    )
    methods = block.get_mmcif_category("_exptl.").get("method") or []
    row["method"] = ";".join(sorted({str(m).strip() for m in methods if m})) or None
    resolution, source = _resolution(block)
    row["resolution"] = resolution
    row["resolution_source"] = source

    poly = block.get_mmcif_category("_entity_poly.")
    entity_ids = poly.get("entity_id") or []
    types = poly.get("type") or []
    strands = poly.get("pdbx_strand_id") or []
    seqs = poly.get("pdbx_seq_one_letter_code_can") or []
    protein_entities: list[str] = []
    lengths: list[int] = []
    strand_ids: list[str] = []
    for i, entity_id in enumerate(entity_ids):
        etype = str(types[i]).strip().lower() if i < len(types) and types[i] else ""
        if etype not in PROTEIN_ENTITY_TYPES:
            continue
        seq = str(seqs[i]) if i < len(seqs) and seqs[i] else ""
        seq = "".join(seq.split())
        strand = str(strands[i]).strip() if i < len(strands) and strands[i] else ""
        protein_entities.append(str(entity_id).strip())
        lengths.append(len(seq))
        strand_ids.append(strand)
    row["protein_entity_ids"] = protein_entities
    row["protein_entity_lengths"] = lengths
    row["protein_strand_ids"] = strand_ids
    row["n_protein_entities"] = len(protein_entities)

    # label_asym_id -> entity_id, so we can count protein chains in the ASU
    # and in assembly 1 (both are expressed in label_asym_id space).
    asym = block.get_mmcif_category("_struct_asym.")
    asym_ids = asym.get("id") or []
    asym_entities = asym.get("entity_id") or []
    protein_entity_set = set(protein_entities)
    protein_asyms = {
        str(asym_ids[i]).strip()
        for i in range(min(len(asym_ids), len(asym_entities)))
        if asym_entities[i] is not None
        and str(asym_entities[i]).strip() in protein_entity_set
    }
    row["n_protein_asym_chains"] = len(protein_asyms)

    distinct, built = _assembly_protein_chains(block, "1", protein_asyms)
    row["n_assembly1_protein_asyms"] = distinct
    row["n_assembly1_protein_chains"] = built
    return row


SCHEMA = pa.schema([
    ("pdb_id", pa.string()),
    ("path", pa.string()),
    ("release_date", pa.string()),
    ("deposition_date", pa.string()),
    ("method", pa.string()),
    ("resolution", pa.float64()),
    ("resolution_source", pa.string()),
    ("n_protein_entities", pa.int32()),
    ("n_protein_asym_chains", pa.int32()),
    ("protein_entity_ids", pa.list_(pa.string())),
    ("protein_entity_lengths", pa.list_(pa.int32())),
    ("protein_strand_ids", pa.list_(pa.string())),
    ("n_assembly1_protein_asyms", pa.int32()),
    ("n_assembly1_protein_chains", pa.int32()),
    ("error", pa.string()),
])


def iter_cif_paths(mirror: Path) -> Iterator[str]:
    for entry in sorted(os.scandir(mirror), key=lambda e: e.name):
        if entry.is_file() and entry.name.endswith((".cif", ".cif.gz")):
            yield entry.path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, default=Path("/data/tim/af3-db/mmcif_files"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/exp222_pdb_curation/metadata/entries.parquet"),
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--limit", type=int, default=None, help="scan only the first N files (smoke test)")
    args = parser.parse_args(argv)

    paths = list(iter_cif_paths(args.mirror))
    if args.limit:
        paths = paths[: args.limit]
    print(f"scanning {len(paths)} mmCIF files with {args.workers} workers", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, SCHEMA, compression="zstd")
    started = time.time()
    batch: list[dict[str, Any]] = []
    done = 0
    try:
        with Pool(args.workers) as pool:
            for row in pool.imap_unordered(scan_one, paths, chunksize=64):
                batch.append(row)
                done += 1
                if len(batch) >= 20_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
                if done % 20_000 == 0:
                    rate = done / (time.time() - started)
                    eta = (len(paths) - done) / rate
                    print(
                        f"  {done}/{len(paths)}  {rate:.0f} entries/s  eta {eta/60:.1f} min",
                        flush=True,
                    )
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    finally:
        writer.close()

    elapsed = time.time() - started
    print(f"wrote {args.out} ({done} rows) in {elapsed/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
