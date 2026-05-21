# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build per-protein Protenix job JSONs + cache GT biological-assembly mmCIFs.

Reads FoldBench's ``targets/monomer_protein.csv`` (one row per monomer,
format ``<pdb>-assembly1,<chain>``), takes the first ``--n`` rows, fetches
the biological-assembly mmCIF from RCSB, extracts the canonical
one-letter sequence (including unresolved residues — that's what
``_entity_poly.pdbx_seq_one_letter_code_can`` gives us) for the named
chain's entity, and writes:

    <out>/jobs/<pdb>_<chain>.json       # Protenix-format job
    <out>/gt/<pdb>_<chain>.cif          # the assembly CIF as-is
    <out>/manifest.csv                  # pdb_id, chain_id, n_residues, seq_path

The job JSON has no ``pairedMsaPath`` / ``unpairedMsaPath`` fields — Modal-
side code adds them for MSA mode (pointing at pre-computed a3m files on a
Volume) and leaves them off for single-sequence mode.

Sequences come from the entity_poly block (canonical, includes
unresolved residues) and *not* from atom_site (which would only give us
residues with experimentally determined coordinates) — the user
explicitly asked for the complete sequence so downstream MSA / Protenix
behavior matches what an in-the-wild user would do.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import gemmi
import requests


FOLDBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/BEAM-Labs/FoldBench/main/targets/monomer_protein.csv"
)
RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb}-assembly{n}.cif"


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MonomerTarget:
    """One row of FoldBench's monomer_protein.csv."""

    pdb_id: str        # e.g. "5sbj" (the bare PDB ID; the CSV's "-assembly1" suffix is split off)
    assembly: int      # e.g. 1
    chain_id: str      # e.g. "A"

    @property
    def stem(self) -> str:
        """Slug used in output filenames + the Protenix ``name`` field."""
        return f"{self.pdb_id}_{self.chain_id}"


# --------------------------------------------------------------------------
# FoldBench CSV
# --------------------------------------------------------------------------


def fetch_foldbench_csv(cache: Path) -> Path:
    """Download FoldBench's monomer CSV to ``cache`` if not already there."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache
    r = requests.get(FOLDBENCH_CSV_URL, timeout=30)
    r.raise_for_status()
    cache.write_text(r.text)
    return cache


def parse_foldbench_csv(csv_path: Path) -> list[MonomerTarget]:
    """Parse monomer_protein.csv into ``MonomerTarget``s."""
    out: list[MonomerTarget] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_pdb = row["pdb_id"].strip()
            chain_id = row["chain_id"].strip()
            # CSV format: "5sbj-assembly1" or just "5sbj" (we accept both).
            if "-assembly" in raw_pdb:
                pdb_id, _, suf = raw_pdb.partition("-assembly")
                assembly = int(suf) if suf.isdigit() else 1
            else:
                pdb_id, assembly = raw_pdb, 1
            out.append(MonomerTarget(pdb_id=pdb_id.lower(), assembly=assembly, chain_id=chain_id))
    return out


# --------------------------------------------------------------------------
# RCSB assembly mmCIF
# --------------------------------------------------------------------------


def fetch_assembly_cif(target: MonomerTarget, cache_dir: Path) -> Path:
    """Download the biological-assembly mmCIF for ``target`` into ``cache_dir``.

    Returns the local path. Idempotent: if the file already exists, we
    don't re-download.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{target.stem}.cif"
    if out.exists():
        return out
    url = RCSB_ASSEMBLY_URL.format(pdb=target.pdb_id, n=target.assembly)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


# --------------------------------------------------------------------------
# Sequence extraction
# --------------------------------------------------------------------------


def extract_canonical_sequence(cif_path: Path, chain_id: str) -> str:
    """Return the canonical one-letter sequence for the protein chain.

    Pulls from ``_entity_poly.pdbx_seq_one_letter_code_can`` (parsed via
    gemmi's cif reader, not the higher-level structure API), so
    unresolved residues are included AND modified residues are mapped
    to their canonical parent (e.g. modified-cysteine 3-letter code
    "85L" maps to "C"). The structure API's ``Entity.full_sequence``
    drops the modified-residue mapping for non-tabulated residues and
    silently returns " " (which we'd render as X), so we go to the raw
    CIF field instead.

    These are FoldBench monomers — biological assemblies with a single
    polypeptide entity. If we ever see >1 L-peptide entity, fail loudly.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    rows = list(block.find(
        "_entity_poly.",
        ["entity_id", "type", "pdbx_seq_one_letter_code_can"],
    ))
    peptide_rows = [r for r in rows if "polypeptide(L)" in r[1]]
    if not peptide_rows:
        raise ValueError(f"No polypeptide(L) entity in {cif_path}")
    if len(peptide_rows) > 1:
        ids = [r[0] for r in peptide_rows]
        raise ValueError(
            f"{cif_path} has {len(peptide_rows)} polypeptide(L) entities "
            f"({ids}); row says chain {chain_id!r} but this is not a monomer."
        )
    # Use Row.str(N), not raw [N] indexing: the former strips the
    # semicolon delimiters and surrounding quotes that mmCIF uses for
    # multi-line text fields, the latter leaves them in. Then strip
    # whitespace + newlines for the canonical one-letter form.
    return "".join(peptide_rows[0].str(2).split())


# --------------------------------------------------------------------------
# Protenix JSON
# --------------------------------------------------------------------------


def build_protenix_job(target: MonomerTarget, sequence: str) -> dict:
    """Build one Protenix v2-format job dict for a single protein chain.

    No MSA/template paths — Modal-side code adds those for MSA mode and
    leaves them off for single-sequence mode.
    See https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md
    """
    return {
        "name": target.stem,
        "sequences": [
            {
                "proteinChain": {
                    "sequence": sequence,
                    "count": 1,
                }
            }
        ],
        "covalent_bonds": [],
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def prepare(
    *,
    out_dir: Path,
    n: int | None,
    foldbench_csv_cache: Path,
) -> Path:
    """Top-level: fetch + parse the CSV, then build inputs for the first ``n`` rows.

    Returns the path to ``manifest.csv``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jobs").mkdir(exist_ok=True)
    (out_dir / "gt").mkdir(exist_ok=True)

    csv_path = fetch_foldbench_csv(foldbench_csv_cache)
    targets = parse_foldbench_csv(csv_path)
    if n is not None:
        targets = targets[:n]

    manifest_rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] {target.stem}: fetching assembly{target.assembly} ...", flush=True)
        cif = fetch_assembly_cif(target, out_dir / "gt")
        sequence = extract_canonical_sequence(cif, target.chain_id)
        job = build_protenix_job(target, sequence)
        (out_dir / "jobs" / f"{target.stem}.json").write_text(json.dumps([job], indent=2))
        manifest_rows.append({
            "pdb_id": target.pdb_id,
            "chain_id": target.chain_id,
            "assembly": target.assembly,
            "stem": target.stem,
            "n_residues": len(sequence),
            "gt_cif": str(cif.relative_to(out_dir)),
            "job_json": f"jobs/{target.stem}.json",
        })

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path} ({len(manifest_rows)} targets)")
    return manifest_path


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Wire the ``prepare-inputs`` subcommand into ``cli.py``'s parser."""
    p = subparsers.add_parser(
        "prepare-inputs",
        help="Build per-protein Protenix JSON jobs + cache GT assembly mmCIFs.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output dir (e.g. inputs/).")
    p.add_argument("--n", type=int, default=None, help="Take first N from the CSV (default: all).")
    p.add_argument(
        "--foldbench-csv-cache",
        type=Path,
        default=Path("/tmp/foldbench_monomer_protein.csv"),
        help="Where to cache the downloaded FoldBench CSV.",
    )
    p.set_defaults(func=lambda args: prepare(
        out_dir=args.out, n=args.n, foldbench_csv_cache=args.foldbench_csv_cache,
    ))
