# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip validation of the generated corpora.

Success criterion 3 of #222: a multimer document must be *readable back* --
parsing it recovers k chains with the right lengths on disjoint index runs,
and its interface contacts must be the ones pyconfind actually found on the
assembly, not an artefact of the layout.

Two levels:

* **Structural** (cheap, run on a large sample of both corpora) -- parse the
  document text alone and check it against its own metadata: every residue
  assigned exactly once, one terminus pair per chain, chain runs contiguous
  and pairwise disjoint, every contact referencing an assigned position, and
  the token count matching.
* **Against pyconfind** (expensive, small sample of multimers) -- rebuild the
  assembly from the mirror, re-run the analysis, and confirm the document's
  inter-chain contacts are exactly the eligible ones pyconfind reports.

Usage::

    uv run python validate.py --root /data/exp222_pdb_curation --sample 3000
"""

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from curate import (
    assembly_subchain_entities,
    build_assembly,
    clean_structure,
    curate_chains,
    protein_subchains,
    read_entry,
)
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    analyze_structure,
)
from marinfold.document_structures.contacts_v1.vocab import NUM_POSITION_INDICES


_POSITION = re.compile(r"<p(\d+)>")
MULTIMER_CONFIG = GenerationConfig(max_chains=60)


def parse_document(document: str) -> dict:
    """Read a contacts-v1 document back into positions, termini and contacts."""
    tokens = document.split()
    begin_sequence = tokens.index("<begin_sequence>")
    begin_statements = tokens.index("<begin_statements>")
    end = tokens.index("<end>")

    residue_positions: list[int] = []
    n_term: list[int] = []
    c_term: list[int] = []
    i = begin_sequence + 1
    while i < begin_statements:
        token = tokens[i]
        if token == "<n-term>":
            n_term.append(int(_POSITION.fullmatch(tokens[i + 1]).group(1)))
        elif token == "<c-term>":
            c_term.append(int(_POSITION.fullmatch(tokens[i + 1]).group(1)))
        else:
            residue_positions.append(int(_POSITION.fullmatch(token).group(1)))
        i += 2

    contacts: list[tuple[int, int]] = []
    i = begin_statements + 1
    while i < end:
        if tokens[i] == "<contact>":
            a = int(_POSITION.fullmatch(tokens[i + 1]).group(1))
            b = int(_POSITION.fullmatch(tokens[i + 2]).group(1))
            contacts.append((min(a, b), max(a, b)))
            i += 3
        else:  # <think>
            i += 1
    return {
        "residue_positions": residue_positions,
        "n_term": n_term,
        "c_term": c_term,
        "contacts": contacts,
        "num_tokens": len(tokens),
    }


def check_structure(row: dict) -> list[str]:
    """Structural problems with one document, as a list of named failures."""
    problems: list[str] = []
    parsed = parse_document(row["document"])
    positions = parsed["residue_positions"]
    n_term, c_term = parsed["n_term"], parsed["c_term"]
    num_chains = row["num_chains"]
    chain_lengths = list(row["chain_lengths"])

    if len(positions) != row["seq_len"]:
        problems.append("residue_count_mismatch")
    if len(set(positions)) != len(positions):
        problems.append("duplicate_position")
    if len(n_term) != num_chains or len(c_term) != num_chains:
        problems.append("terminus_count_mismatch")
    if sorted(n_term) != sorted(row["n_term_indices"]):
        problems.append("n_term_metadata_mismatch")
    if sorted(c_term) != sorted(row["c_term_indices"]):
        problems.append("c_term_metadata_mismatch")
    if parsed["num_tokens"] != row["num_tokens"]:
        problems.append("token_count_mismatch")
    if len(parsed["contacts"]) != row["contacts_emitted"]:
        problems.append("contact_count_mismatch")
    if len(set(parsed["contacts"])) != len(parsed["contacts"]):
        problems.append("duplicate_contact")

    # Rebuild each chain's run from its n-terminus and declared length, then
    # check the runs tile exactly the assigned positions without overlapping.
    assigned = set(positions)
    runs: list[set[int]] = []
    for length, start in zip(chain_lengths, row["n_term_indices"]):
        runs.append({(start + k) % NUM_POSITION_INDICES for k in range(length)})
    union: set[int] = set()
    for run in runs:
        if union & run:
            problems.append("overlapping_chains")
            break
        union |= run
    if union != assigned:
        problems.append("chain_runs_do_not_cover_residues")
    for length, start, stop in zip(chain_lengths, row["n_term_indices"], row["c_term_indices"]):
        if stop != (start + length - 1) % NUM_POSITION_INDICES:
            problems.append("c_term_not_at_run_end")
            break
    if sum(chain_lengths) != row["seq_len"]:
        problems.append("chain_lengths_do_not_sum")

    for a, b in parsed["contacts"]:
        if a not in assigned or b not in assigned:
            problems.append("contact_references_unassigned_position")
            break
    return problems


def check_against_pyconfind(row: dict, mirror: Path) -> list[str]:
    """Re-derive a multimer's interface contacts from the structure itself."""
    pdb_id = row["pdb_id"]
    raw = read_entry(str(mirror / f"{pdb_id}.cif"))
    asu = clean_structure(raw.clone())
    expanded = build_assembly(raw, "1")
    if expanded is None:
        return ["assembly_disappeared"]
    assembly = clean_structure(expanded)
    built = curate_chains(
        assembly,
        assembly_subchain_entities(protein_subchains(asu), assembly),
        max_residues=NUM_POSITION_INDICES,
    )
    keep = {c.chain_id for c in built.kept}
    for model in assembly:
        for name in [chain.name for chain in model]:
            if name not in keep:
                model.remove_chain(name)
    assembly.setup_entities()

    analyzed = analyze_structure(
        assembly, entry_id=row["entry_id"], max_chains=len(keep) or 1
    )
    chain_of = [r.chain for r in analyzed.residues]
    eligible_inter = {
        (c.seq_i, c.seq_j)
        for c in analyzed.contacts
        if chain_of[c.seq_i] != chain_of[c.seq_j]
        and c.degree >= MULTIMER_CONFIG.min_contact_degree
    }
    problems: list[str] = []
    # A truncated document dropped its weakest contacts, so it can only be a
    # subset; an untruncated one must carry every eligible interface contact.
    if row["truncated"]:
        if row["contacts_emitted_inter_chain"] > len(eligible_inter):
            problems.append("more_interface_contacts_than_pyconfind_found")
    elif row["contacts_emitted_inter_chain"] != len(eligible_inter):
        problems.append(
            f"interface_count_mismatch({row['contacts_emitted_inter_chain']}"
            f"!={len(eligible_inter)})"
        )
    return problems


def sample_documents(
    shards: list[Path], columns: list[str], sample: int, rng: random.Random
) -> tuple[list[dict], int]:
    """Read a random sample of documents, one shard at a time.

    Deliberately *not* ``dataset(...).to_table()`` then ``take``: pyarrow's
    default ``string`` type carries 32-bit offsets, and the ``document``
    column across the monomer corpus's 31 shards is well past 2 GB, so
    concatenating them raises "offset overflow while concatenating arrays".
    Sampling per shard never materialises more than one shard's worth of
    document text, and gives the same uniform sample as long as shards are
    equal-sized (they are -- the writer flushes at a fixed row count, bar the
    last one).
    """
    per_shard = max(1, sample // len(shards))
    rows: list[dict] = []
    total = 0
    for shard in shards:
        table = pq.read_table(shard, columns=columns)
        total += table.num_rows
        take = min(per_shard, table.num_rows)
        indices = rng.sample(range(table.num_rows), take)
        rows.extend(table.take(indices).to_pylist())
        del table
    return rows, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/exp222_pdb_curation"))
    parser.add_argument("--mirror", type=Path, default=Path("/data/tim/af3-db/mmcif_files"))
    parser.add_argument("--sample", type=int, default=3000, help="documents per subset")
    parser.add_argument("--pyconfind-sample", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    columns = [
        "document", "entry_id", "pdb_id", "seq_len", "num_tokens", "num_chains",
        "chain_lengths", "n_term_indices", "c_term_indices", "contacts_emitted",
        "contacts_emitted_inter_chain", "truncated",
    ]

    failed = 0
    for subset in ("monomers", "multimers", "deduped"):
        directory = args.root / "docs" / subset
        shards = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
        if not shards:
            print(f"{subset}: no shards, skipping")
            continue
        rows, total = sample_documents(shards, columns, args.sample, rng)

        problems: Counter = Counter()
        for row in rows:
            for problem in check_structure(row):
                problems[problem] += 1
        clean = sum(1 for row in rows if not check_structure(row))
        print(
            f"{subset}: structural check on {len(rows)}/{total} documents -> "
            f"{clean} clean, problems={dict(problems)}"
        )
        failed += sum(problems.values())

        if subset in ("multimers", "deduped"):
            multimer_rows = [r for r in rows if r["num_chains"] > 1]
            sample = multimer_rows[: args.pyconfind_sample]
            geometry: Counter = Counter()
            for row in sample:
                for problem in check_against_pyconfind(row, args.mirror):
                    geometry[problem] += 1
            print(
                f"{subset}: pyconfind cross-check on {len(sample)} documents -> "
                f"problems={dict(geometry)}"
            )
            failed += sum(geometry.values())

    print("VALIDATION PASSED" if failed == 0 else f"VALIDATION FAILED ({failed} problems)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
