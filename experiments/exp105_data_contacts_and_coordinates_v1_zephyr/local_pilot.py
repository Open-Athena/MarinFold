# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local pilot — generate a handful of documents off-cluster to size the run.

The selection manifest points at AFDB structures in a requester-pays GCS
bucket that is only reachable from the Iris cluster's service account, so a
truly local pass fetches the *same* AlphaFold models from EBI's public
HTTPS endpoint instead (``AF-<acc>-F1-model_v4.cif``). It generates one
contacts-and-coordinates-v1 document per accession, prints per-document
token / size statistics and a corpus projection, and dumps a couple of
example documents to eyeball before committing to a full run.

This is the "initial pass on a small subset locally to estimate costs and
data size" from issue #105 — not a substitute for the on-cluster iris smoke
(which measures real per-doc wall-clock and validates the requester-pays
fetch), but enough to look at real documents and project token volume.

Usage::

    uv run python local_pilot.py --out pilot.parquet
    uv run python local_pilot.py --accessions my_accessions.txt --projected-docs 4200000
"""

import argparse
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_and_coordinates_v1 import (
    CONTEXT_LENGTH,
    GenerationConfig,
    generate_document,
)

import generate_rows

_EBI_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif"

# A size-spread default sample of UniProt accessions (small → large single
# chains). Not the selection manifest itself — a representative stand-in for
# a local token/size estimate. Override with --accessions.
_DEFAULT_ACCESSIONS = [
    "P0DTC2", "P59594", "P0DTD1", "P01308", "P69905", "P68871",
    "P00533", "P04637", "P38398", "P00766", "P02768", "P0DP23",
    "P42212", "P0AEG4", "P0A7Y4", "P60484", "P01112", "P02185",
    "P0A6F5", "P06654", "P00698", "P00918", "P0A9P0", "P0AA25",
]


def _fetch_cif(accession: str, *, timeout: float = 30.0) -> str | None:
    """Fetch one AlphaFold mmCIF from the EBI public endpoint (None on failure)."""
    url = _EBI_URL.format(acc=accession)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — a missing model is fine, just skip
        print(f"  ! fetch failed for {accession}: {exc}", file=sys.stderr)
        return None


def run_pilot(
    accessions: list[str],
    *,
    config: GenerationConfig,
    context_length: int,
    projected_docs: int,
) -> list[dict[str, Any]]:
    """Generate one document per accession; print stats + a corpus projection."""
    rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    doc_bytes: list[int] = []
    for acc in accessions:
        cif = _fetch_cif(acc)
        if cif is None:
            continue
        entry_id = f"AF-{acc}-F1-model_v4"
        try:
            structure = generate_rows.structure_from_cif(cif, entry_id=entry_id)
            result = generate_document(
                structure, entry_id=entry_id,
                context_length=context_length, config=config,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"  ! generate failed for {acc}: {exc}", file=sys.stderr)
            continue
        if result is None:
            print(f"  - skipped {acc} (unserializable)", file=sys.stderr)
            continue
        row = result.metadata_row()
        rows.append(row)
        token_counts.append(result.num_tokens)
        doc_bytes.append(len(result.document.encode("utf-8")))
        print(
            f"  {entry_id}: residues={result.seq_len} "
            f"tokens={result.num_tokens} events={result.num_events} "
            f"distinct_atoms={result.num_distinct_atoms_mentioned} "
            f"truncated={result.truncated}"
        )

    if not rows:
        print("No documents generated — check network access to EBI.", file=sys.stderr)
        return rows

    mean_tokens = statistics.mean(token_counts)
    mean_bytes = statistics.mean(doc_bytes)
    print("\n=== pilot summary ===")
    print(f"  documents:        {len(rows)}")
    print(f"  tokens/doc:       mean={mean_tokens:,.0f}  "
          f"median={statistics.median(token_counts):,.0f}  "
          f"min={min(token_counts):,}  max={max(token_counts):,}")
    print(f"  raw doc bytes:    mean={mean_bytes:,.0f}")
    truncated = sum(1 for r in rows if r["truncated"])
    print(f"  truncated:        {truncated}/{len(rows)}")
    print(f"\n=== projection to {projected_docs:,} docs ===")
    print(f"  total tokens:     ~{mean_tokens * projected_docs / 1e9:,.1f} B")
    print(f"  raw text size:    ~{mean_bytes * projected_docs / 1e12:,.2f} TB "
          f"(parquet with zstd is typically ~4-8x smaller)")

    # Show an example document (truncated for readability).
    example = rows[0]["document"].split()
    print("\n=== example document (first 60 tokens) ===")
    print(" ".join(example[:60]) + " …")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accessions", type=Path, default=None,
                        help="Newline-delimited UniProt accessions "
                             "(default: a built-in size-spread sample).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional parquet to write the generated rows to.")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                        help=f"Token budget per document (default {CONTEXT_LENGTH}).")
    parser.add_argument("--projected-docs", type=int, default=4_200_000,
                        help="Corpus size to project token/byte totals to.")
    args = parser.parse_args(argv)

    if args.accessions is not None:
        accessions = [ln.strip() for ln in args.accessions.read_text().splitlines()
                      if ln.strip()]
    else:
        accessions = _DEFAULT_ACCESSIONS

    print(f"Fetching + generating for {len(accessions)} accessions …")
    rows = run_pilot(
        accessions,
        config=GenerationConfig(),
        context_length=args.context_length,
        projected_docs=args.projected_docs,
    )
    if args.out is not None and rows:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, args.out)
        print(f"\nwrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
