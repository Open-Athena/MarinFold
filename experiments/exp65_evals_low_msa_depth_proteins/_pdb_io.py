# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared download + manifest helpers for the exp65 fetch scripts.

The three fetch scripts (``fetch_denovo_pdb.py``, ``fetch_casp_fm.py``,
``fetch_cameo_hard.py``) pull structures from different sources but share
two concerns: getting a file over HTTP idempotently, and emitting a single
common manifest schema so the survivors can later be cross-labelled with the
exp41 Foldseek fold-novelty verdict and the ``msa_depth.py`` Neff tier (the
2-D label in ``notes/low-msa-eval-curation.md`` section 6).

The download idiom mirrors
``experiments/exp12_data_protenix_foldbench_monomers/prepare_inputs.py``
(``requests.get`` + ``raise_for_status`` + an ``.exists()`` skip), so a
re-run is a no-op for files already on disk.
"""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import gemmi
import requests

RCSB_FILE_URL = "https://files.rcsb.org/download/{pdb}.cif"

# Shared manifest columns. ``stem`` is ``<id>_<chain>`` (or just the target
# id when a single-chain target has no obvious chain letter) so these rows
# join onto exp12/exp41 CSVs and the future Foldseek + Neff labels. Empty
# string for fields a given source can't fill (e.g. CASP/CAMEO resolution).
MANIFEST_FIELDS = (
    "source",        # denovo_pdb | casp14_fm | casp15_fm | cameo_hard
    "stem",          # <id>_<chain>, the cross-experiment join key
    "pdb_id",        # source structure id (PDB id, CASP target, CAMEO target)
    "chain",         # chain id(s) kept, comma-separated
    "length",        # residue count (best-effort; -1 if unknown)
    "resolution",    # Angstrom, or "" if not applicable
    "deposit_date",  # ISO date, or "" if not applicable
    "category",      # source label: "DE NOVO PROTEIN" | "FM" | "FM/TBM" | "hard"
    "novelty_axis",  # which novelty axis this source probes (free text)
    "local_path",    # path to the downloaded structure, relative to the exp dir
)


@dataclass
class ManifestRow:
    source: str
    stem: str
    pdb_id: str
    chain: str = ""
    length: int = -1
    resolution: str = ""
    deposit_date: str = ""
    category: str = ""
    novelty_axis: str = ""
    local_path: str = ""


# Keep the dataclass and the column tuple in lockstep; a drift here would
# silently write a malformed manifest.
assert tuple(f.name for f in fields(ManifestRow)) == MANIFEST_FIELDS


def http_get(url: str, *, timeout: int = 60, **kwargs) -> requests.Response:
    """``requests.get`` with a default timeout, raising on HTTP error."""
    r = requests.get(url, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r


def download_to(url: str, out_path: Path, *, timeout: int = 120) -> Path:
    """Download ``url`` to ``out_path`` (idempotent: skip if it exists)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    r = http_get(url, timeout=timeout)
    out_path.write_bytes(r.content)
    return out_path


def download_cif(pdb_id: str, out_dir: Path, *, timeout: int = 120) -> Path:
    """Download the asymmetric-unit mmCIF for ``pdb_id`` from RCSB.

    Returns the local path. Idempotent. ``pdb_id`` is lower-cased to match
    RCSB's file-server convention.
    """
    pdb_id = pdb_id.lower()
    return download_to(
        RCSB_FILE_URL.format(pdb=pdb_id), out_dir / f"{pdb_id}.cif", timeout=timeout
    )


def read_structure(path: Path) -> gemmi.Structure:
    """Read a structure (.cif/.pdb, optionally .gz) via gemmi."""
    return gemmi.read_structure(str(path))


def polymer_chains(path: Path) -> dict[str, int]:
    """Map chain id -> polymer residue count for the first model.

    Counts only polymer (peptide/nucleotide) residues, so ligands and
    waters don't inflate the length. ``setup_entities()`` is called first so
    raw CASP/CAMEO PDBs with blank chain ids (where ``get_polymer()`` would
    otherwise return nothing) are classified correctly; a CA-atom count is the
    last-resort fallback for files gemmi still can't parse as a polymer. Empty
    dict if the file has no model.
    """
    st = read_structure(path)
    if not st:
        return {}
    st.setup_entities()
    out: dict[str, int] = {}
    for chain in st[0]:
        n = len(chain.get_polymer())
        if not n:
            # Fallback: count residues bearing a CA atom (protein residues).
            n = sum(1 for res in chain if res.find_atom("CA", "*") is not None)
        if n:
            out[chain.name] = n
    return out


def count_residues(path: Path) -> int:
    """Total polymer residue count across all chains in the first model."""
    return sum(polymer_chains(path).values())


def write_manifest(rows: list[ManifestRow], out_path: Path) -> Path:
    """Write ``rows`` to ``out_path`` as CSV with the shared schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return out_path
