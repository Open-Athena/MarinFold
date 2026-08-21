# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a query FASTA for *all* of FoldBench, not just the 100 monomers we score.

The pinned v1 reference is the 554-protein benchmark we actually report on, and
100 of those come from FoldBench's monomer set. But FoldBench is much larger
than its monomers — protein-protein, antibody-antigen, protein-peptide,
protein-ligand, protein-DNA and protein-RNA interface tasks all carry protein
chains too. If we ever want to report on any of those, their chains have to be
out of the training corpus as well, and that is a strictly larger reference than
the one Stage 1 pinned.

This assembles that larger reference. ``FoldBench/targets/*.csv`` name targets as
``(pdb_id, chain_id)`` with an explicit chain type per side, so protein and
peptide chains can be selected without guessing; sequences come from RCSB's
per-entry FASTA endpoint (entity sequences, i.e. SEQRES — the right thing for a
homology search, since unresolved residues still exist in the protein).

Assembly ids (``8wmt-assembly1``) and symmetry-copy chain labels (``A-2``) are
reduced to their base entry and chain, because a symmetry copy has the same
sequence as the chain it was copied from.

    uv run python foldbench_reference.py --out data/reference/foldbench_all_queries.fasta
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_TARGETS_DIR = Path("/home/bizon/git/FoldBench/targets")

#: Which chain types count as protein for our purposes. Peptides are short
#: protein chains and contacts-v1 can be trained to leak them just the same.
PROTEIN_TYPES = frozenset({"protein", "peptide"})

#: ``targets/*.csv`` files and how to read a (chain_id, chain_type) pair out of
#: each row. ``monomer_protein`` has no type column — every row is protein.
INTERFACE_FILES = (
    "interface_protein_protein.csv",
    "interface_antibody_antigen.csv",
    "interface_protein_peptide.csv",
    "interface_protein_dna.csv",
    "interface_protein_rna.csv",
)
LIGAND_FILE = "interface_protein_ligand.csv"
MONOMER_FILE = "monomer_protein.csv"

RCSB_FASTA = "https://www.rcsb.org/fasta/entry/{ids}"

#: ``>8WMT_1|Chains A, B|desc|organism`` or ``|Chain A|`` or
#: ``|Chains A[auth AAA], B|``. Both the label and the auth id are indexed,
#: because FoldBench's assembly chain labels can be either.
_CHAIN_TOKEN = re.compile(r"\s*([A-Za-z0-9]+)(?:\[auth\s+([A-Za-z0-9]+)\])?\s*")


def base_entry(pdb_id: str) -> str:
    """``8wmt-assembly1`` -> ``8WMT``."""
    return pdb_id.split("-")[0].upper()


def base_chain(chain_id: str) -> str:
    """``A-2`` -> ``A`` — a symmetry copy has its source chain's sequence."""
    return chain_id.split("-")[0]


def collect_chains(targets_dir: Path) -> dict[str, set[str]]:
    """entry -> set of protein chain ids, over every FoldBench task file."""
    wanted: defaultdict[str, set[str]] = defaultdict(set)

    def add(pdb_id: str, chain_id: str) -> None:
        if chain_id:
            wanted[base_entry(pdb_id)].add(base_chain(chain_id))

    monomers = targets_dir / MONOMER_FILE
    with monomers.open() as fh:
        for row in csv.DictReader(fh):
            add(row["pdb_id"], row["chain_id"])

    for name in INTERFACE_FILES:
        path = targets_dir / name
        with path.open() as fh:
            for row in csv.DictReader(fh):
                for side in ("1", "2"):
                    if row[f"interface_chain_type_{side}"] in PROTEIN_TYPES:
                        add(row["pdb_id"], row[f"interface_chain_id_{side}"])

    with (targets_dir / LIGAND_FILE).open() as fh:
        for row in csv.DictReader(fh):
            for side in ("1", "2"):
                if row[f"native_chain_type_{side}"] in PROTEIN_TYPES:
                    add(row["pdb_id"], row[f"native_chain_id_{side}"])
    return dict(wanted)


def parse_fasta(text: str) -> dict[str, dict[str, str]]:
    """RCSB entry FASTA -> ``{entry: {chain_id: sequence}}``.

    Every chain listed on an entity's header maps to that entity's sequence,
    under both its label and (where given) its auth id.
    """
    entries: defaultdict[str, dict[str, str]] = defaultdict(dict)
    header: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if header is None:
            return
        fields = header.split("|")
        entry = fields[0].split("_")[0].upper()
        sequence = "".join(parts).strip().upper()
        if len(fields) < 2 or not sequence:
            return
        for token in fields[1].removeprefix("Chains").removeprefix("Chain").split(","):
            match = _CHAIN_TOKEN.fullmatch(token)
            if not match:
                continue
            for chain in (match.group(1), match.group(2)):
                if chain:
                    entries[entry][chain] = sequence

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header, parts = line[1:], []
        elif header is not None:
            parts.append(line.strip())
    flush()
    return dict(entries)


def fetch_batch(entries: tuple[str, ...], *, retries: int = 4) -> dict[str, dict[str, str]]:
    """One batched RCSB request, with a plain backoff on transport errors."""
    url = RCSB_FASTA.format(ids=",".join(entries))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return parse_fasta(response.read().decode())
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise SystemExit(f"RCSB fetch failed for {entries[:3]}...: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS_DIR)
    ap.add_argument("--out", type=Path,
                    default=HERE / "data/reference/foldbench_all_queries.fasta")
    ap.add_argument("--provenance-out", type=Path,
                    default=HERE / "data/reference/foldbench_all.provenance.json")
    ap.add_argument("--batch", type=int, default=50, help="entries per RCSB request")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=10,
                    help="sequences shorter than this cannot carry an alignment and "
                         "mmseqs cannot index them")
    args = ap.parse_args()

    wanted = collect_chains(args.targets_dir)
    n_chains = sum(len(chains) for chains in wanted.values())
    print(f"[foldbench] {len(wanted)} entries, {n_chains} protein chains requested",
          flush=True)

    ids = sorted(wanted)
    batches = [tuple(ids[i:i + args.batch]) for i in range(0, len(ids), args.batch)]
    sequences: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(fetch_batch, batches), 1):
            sequences.update(result)
            if done % 5 == 0 or done == len(batches):
                print(f"[foldbench] {done}/{len(batches)} batches", flush=True)

    records: dict[str, str] = {}
    missing: list[str] = []
    skipped_short = 0
    for entry, chains in sorted(wanted.items()):
        for chain in sorted(chains):
            sequence = sequences.get(entry, {}).get(chain)
            if sequence is None:
                missing.append(f"{entry}_{chain}")
                continue
            if len(sequence) < args.min_len:
                skipped_short += 1
                continue
            records[f"foldbench_all__{entry}_{chain}"] = sequence

    # Identical sequences across chains/entries are kept as separate records:
    # mmseqs deduplicates nothing here, and a drop list is a union over queries
    # so duplicates cost search time but cannot change the answer.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f">{k}\n{v}\n" for k, v in sorted(records.items())))
    unique_sequences = len(set(records.values()))
    print(f"[foldbench] {len(records)} chains -> {args.out} "
          f"({unique_sequences} unique sequences)", flush=True)
    if missing:
        print(f"[warn] {len(missing)} chains had no RCSB sequence: {missing[:10]}",
              flush=True)

    args.provenance_out.write_text(
        json.dumps(
            {
                "source": str(args.targets_dir),
                "task_files": [MONOMER_FILE, *INTERFACE_FILES, LIGAND_FILE],
                "protein_chain_types": sorted(PROTEIN_TYPES),
                "n_entries": len(wanted),
                "n_chains_requested": n_chains,
                "n_chains_written": len(records),
                "n_unique_sequences": unique_sequences,
                "n_chains_missing": len(missing),
                "chains_missing": missing,
                "n_skipped_short": skipped_short,
                "min_len": args.min_len,
                "sequence_source": "RCSB entry FASTA (entity/SEQRES sequences)",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[provenance] -> {args.provenance_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
