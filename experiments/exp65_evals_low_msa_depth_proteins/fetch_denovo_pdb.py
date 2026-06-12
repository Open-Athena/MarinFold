# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the PDB ``DE NOVO PROTEIN`` class as low-MSA / novel-fold candidates.

De novo designed proteins are the cleanest "far from training" bucket for a
single-sequence model (``notes/eval-dataset-design.md`` section 7.1,
``notes/low-msa-eval-curation.md`` section 4.1): no natural evolutionary
lineage, so a ColabFold search returns ~just the query (Neff ~= 1), and many
occupy fold space absent from AFDB/UniProt. The bare keyword query returns
~2,007 experimental entries (verified live, June 2026); we filter
server-side to single-chain experimental monomers in a sensible length /
resolution band, optionally after a deposition-date cutoff (the AFDB-snapshot
leakage guard), then download the mmCIFs and write the shared manifest.

This is the first script in the repo to hit the RCSB *search* API
(``search.rcsb.org``); existing code only pulls files from
``files.rcsb.org`` by id. Structures land in ``structures/denovo/``
(gitignored); the manifest lands in ``data/denovo_pdb_manifest.csv``
(committed).
"""

import argparse
import json
from pathlib import Path

import gemmi
import requests

from _pdb_io import (
    ManifestRow,
    count_residues,
    download_cif,
    polymer_chains,
    write_manifest,
)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DENOVO_KEYWORD = "DE NOVO PROTEIN"

HERE = Path(__file__).resolve().parent


def _terminal(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def build_query(
    *,
    min_len: int,
    max_len: int,
    max_resolution: float | None,
    monomer_only: bool,
    after_date: str | None,
) -> dict:
    """Build the compound RCSB search query for de novo monomers.

    AND of: the DE NOVO PROTEIN keyword, an experimental-method gate (so
    computed models are excluded), a sample-sequence-length band, and
    optionally a single-instance (monomer) gate, a resolution ceiling, and a
    deposition-date floor.
    """
    nodes = [
        _terminal(
            "struct_keywords.pdbx_keywords", "contains_phrase", DENOVO_KEYWORD
        ),
        # Exclude AlphaFold/computed entries — we want real experimental GT.
        _terminal(
            "rcsb_entry_info.structure_determination_methodology",
            "exact_match",
            "experimental",
        ),
        _terminal(
            "entity_poly.rcsb_sample_sequence_length",
            "range",
            {"from": min_len, "to": max_len, "include_lower": True, "include_upper": True},
        ),
    ]
    if monomer_only:
        # One deposited polymer chain instance == a true monomer.
        nodes.append(
            _terminal(
                "rcsb_entry_info.deposited_polymer_entity_instance_count",
                "equals",
                1,
            )
        )
    if max_resolution is not None:
        # NOTE: this also drops NMR entries (no resolution). Pass
        # --max-resolution -1 to disable if you want NMR designs.
        nodes.append(
            _terminal(
                "rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution
            )
        )
    if after_date:
        nodes.append(
            _terminal(
                "rcsb_accession_info.deposit_date", "greater", after_date
            )
        )
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }


def search_entry_ids(query: dict) -> list[str]:
    """POST ``query`` to the RCSB search API and return entry ids.

    The search API takes the whole envelope (query + return_type +
    request_options) as the POST body. An empty result set yields HTTP 204
    (no body); we surface that as an empty list rather than letting the JSON
    decode blow up.
    """
    resp = requests.post(SEARCH_URL, json=query, timeout=120)
    if resp.status_code == 204:
        return []
    resp.raise_for_status()
    payload = resp.json()
    return [hit["identifier"] for hit in payload.get("result_set", [])]


def count_only(query: dict) -> int:
    """Return just the total hit count for ``query`` (cheap sanity check)."""
    envelope = json.loads(json.dumps(query))  # shallow copy of the envelope
    envelope["request_options"] = {"return_counts": True}
    resp = requests.post(SEARCH_URL, json=envelope, timeout=120)
    if resp.status_code == 204:
        return 0
    resp.raise_for_status()
    return int(resp.json()["total_count"])


def _deposit_date(cif_path: Path) -> str:
    """Best-effort deposition date from the mmCIF (empty string if absent)."""
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    val = block.find_value("_pdbx_database_status.recvd_initial_deposition_date")
    return (val or "").strip().strip("'\"")


def _resolution(cif_path: Path) -> str:
    res = gemmi.read_structure(str(cif_path)).resolution
    return f"{res:.2f}" if res and res > 0 else ""


def manifest_row(pdb_id: str, cif_path: Path) -> ManifestRow:
    """Build a manifest row from a downloaded de novo entry."""
    chains = polymer_chains(cif_path)
    chain_str = ",".join(sorted(chains))
    # Single-chain monomer -> <pdb>_<chain>; multi-chain -> just the id.
    stem = f"{pdb_id.lower()}_{next(iter(sorted(chains)))}" if len(chains) == 1 else pdb_id.lower()
    return ManifestRow(
        source="denovo_pdb",
        stem=stem,
        pdb_id=pdb_id.lower(),
        chain=chain_str,
        length=count_residues(cif_path),
        resolution=_resolution(cif_path),
        deposit_date=_deposit_date(cif_path),
        category=DENOVO_KEYWORD,
        novelty_axis="designed: no evolutionary homologs (Neff~=1); often novel fold",
        local_path=str(cif_path.relative_to(HERE)),
    )


def run(
    *,
    out_dir: Path,
    manifest_path: Path,
    min_len: int,
    max_len: int,
    max_resolution: float | None,
    monomer_only: bool,
    after_date: str | None,
    limit: int | None,
    download: bool,
) -> list[ManifestRow]:
    query = build_query(
        min_len=min_len,
        max_len=max_len,
        max_resolution=max_resolution,
        monomer_only=monomer_only,
        after_date=after_date,
    )
    ids = search_entry_ids(query)
    print(f"RCSB search: {len(ids)} DE NOVO PROTEIN entries match the filters")
    if limit is not None:
        ids = ids[:limit]
        print(f"  (limited to first {len(ids)} for this run)")

    rows: list[ManifestRow] = []
    if not download:
        # Manifest-only: just record the ids, no structure metadata.
        for pdb_id in ids:
            rows.append(
                ManifestRow(
                    source="denovo_pdb",
                    stem=pdb_id.lower(),
                    pdb_id=pdb_id.lower(),
                    category=DENOVO_KEYWORD,
                    novelty_axis="designed: no evolutionary homologs (Neff~=1); often novel fold",
                )
            )
        write_manifest(rows, manifest_path)
        print(f"Wrote {len(rows)} ids (no download) to {manifest_path}")
        return rows

    for i, pdb_id in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] {pdb_id}: downloading mmCIF ...", flush=True)
        cif = download_cif(pdb_id, out_dir)
        rows.append(manifest_row(pdb_id, cif))
    write_manifest(rows, manifest_path)
    print(f"Wrote {len(rows)} rows to {manifest_path}; structures in {out_dir}/")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "structures" / "denovo",
        help="Where to write downloaded mmCIFs (default: ./structures/denovo/).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "data" / "denovo_pdb_manifest.csv",
        help="Manifest CSV path (default: ./data/denovo_pdb_manifest.csv).",
    )
    p.add_argument("--min-len", type=int, default=40, help="Min sample sequence length.")
    p.add_argument("--max-len", type=int, default=400, help="Max sample sequence length.")
    p.add_argument(
        "--max-resolution",
        type=float,
        default=3.0,
        help="Resolution ceiling in Angstrom (also excludes NMR). Pass -1 to disable.",
    )
    p.add_argument(
        "--no-monomer",
        dest="monomer_only",
        action="store_false",
        help="Keep multi-chain entries too (default: single-instance monomers only).",
    )
    p.add_argument(
        "--after-date",
        default=None,
        help="Keep only entries deposited after this ISO date (AFDB-snapshot leakage guard).",
    )
    p.add_argument("--limit", type=int, default=None, help="Take first N hits (smoke test).")
    p.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Write the id list only; skip mmCIF downloads + structure metadata.",
    )
    p.add_argument(
        "--count-only",
        action="store_true",
        help="Just print the total hit count for the filters and exit.",
    )
    args = p.parse_args()

    max_res = None if args.max_resolution is not None and args.max_resolution < 0 else args.max_resolution

    if args.count_only:
        q = build_query(
            min_len=args.min_len,
            max_len=args.max_len,
            max_resolution=max_res,
            monomer_only=args.monomer_only,
            after_date=args.after_date,
        )
        print(count_only(q))
        return

    run(
        out_dir=args.out,
        manifest_path=args.manifest,
        min_len=args.min_len,
        max_len=args.max_len,
        max_resolution=max_res,
        monomer_only=args.monomer_only,
        after_date=args.after_date,
        limit=args.limit,
        download=args.download,
    )


if __name__ == "__main__":
    main()
