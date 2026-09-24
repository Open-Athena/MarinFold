#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0b — the ground-truth noise floor for the fold-switch contact comparison.

``prepare_inputs.py`` reports a median Jaccard of ~0.52 between the two folds of
a pair. That number means nothing on its own: some of the difference between any
two contact maps is crystallography, not biology. This script measures how much.

The control set is built from the fold-switching entries themselves, with no new
downloads and no curation: **within a single PDB entry, two chains carrying the
same sequence in the same fold**. Same molecule, same crystal, same resolution,
same refinement — so whatever Jaccard they show is the measurement floor, at
exactly the resolutions present in the eval set. The annotated fold-switch chain
pair is excluded, since that pair is the thing being measured.

Each replicate goes through the identical machinery as a real pair (union
reference, mismatch exclusion, contacts_v1's degree and separation cuts), so the
floor and the signal are directly comparable.

Writes ``data/noise_floor.csv``.

    uv run python build_noise_floor.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import gemmi

from prepare_inputs import (
    DATA,
    analyze_chain,
    load_pairs,
    remap_contacts,
    structure_path,
    union_reference,
)

# Entries in this set run from small monomers to ribosome subunits. Analysing
# every chain of a 50-chain assembly buys nothing, so take a bounded sample.
MAX_CHAINS_PER_ENTRY = 8
MAX_REPLICATE_PAIRS_PER_ENTRY = 2

# A replicate needs enough contacts for its Jaccard to be stable.
MIN_CONTACTS = 20


def polymer_chains(path: Path) -> list[str]:
    """Author chain ids of the peptide polymers in the first model."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    while len(st) > 1:
        del st[1]
    names: list[str] = []
    for chain in st[0]:
        try:
            if len(chain.get_polymer()) > 0:
                names.append(chain.name)
        except Exception:  # noqa: BLE001 — non-polymer chains raise; skip them
            continue
    return names


def replicate_rows(pdb: str, excluded: set[frozenset[str]]) -> list[dict]:
    """Same-sequence chain pairs within one entry, scored like a real pair."""
    path = structure_path(pdb)
    chains = polymer_chains(path)[:MAX_CHAINS_PER_ENTRY]
    if len(chains) < 2:
        return []

    analyzed: dict[str, tuple] = {}
    for chain in chains:
        try:
            analyzed[chain] = analyze_chain(path, pdb, chain)
        except Exception as exc:  # noqa: BLE001
            # One unreadable chain must not cost the entry its other pairs;
            # the reason is printed so a systematic failure is still visible.
            print(f"    {pdb}_{chain}: skipped ({type(exc).__name__}: {exc})")

    by_sequence: dict[str, list[str]] = defaultdict(list)
    for chain, (_, obs) in analyzed.items():
        by_sequence[obs].append(chain)

    rows: list[dict] = []
    for obs, group in by_sequence.items():
        if len(group) < 2 or len(obs) < 30:
            continue
        for chain_a, chain_b in itertools.combinations(sorted(group), 2):
            if frozenset({chain_a, chain_b}) in excluded:
                continue  # this is the annotated fold switch, not a replicate
            analyzed_a, obs_a = analyzed[chain_a]
            analyzed_b, obs_b = analyzed[chain_b]
            frame = union_reference(obs_a, obs_b)
            contacts_a, resolved_a = remap_contacts(analyzed_a, frame.map1)
            contacts_b, resolved_b = remap_contacts(analyzed_b, frame.map2)
            common = resolved_a & resolved_b
            set_a = {(i, j) for i, j in contacts_a if i in common and j in common}
            set_b = {(i, j) for i, j in contacts_b if i in common and j in common}
            union = set_a | set_b
            if len(set_a) < MIN_CONTACTS or len(set_b) < MIN_CONTACTS:
                continue
            rows.append({
                "entry": pdb,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "L": len(frame.reference),
                "n_common": len(common),
                "n_contacts_a": len(set_a),
                "n_contacts_b": len(set_b),
                "n_shared": len(set_a & set_b),
                "n_only_a": len(set_a - set_b),
                "n_only_b": len(set_b - set_a),
                "jaccard": len(set_a & set_b) / len(union) if union else 1.0,
            })
            if len(rows) >= MAX_REPLICATE_PAIRS_PER_ENTRY:
                return rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only scan the first N entries")
    args = parser.parse_args()

    pairs = load_pairs()
    # Chain pairs to exclude: the annotated fold switch, for same-entry pairs.
    excluded: dict[str, set[frozenset[str]]] = defaultdict(set)
    for spec in pairs:
        if spec.pdb1 == spec.pdb2:
            excluded[spec.pdb1].add(frozenset({spec.chain1, spec.chain2}))

    entries = sorted({spec.pdb1 for spec in pairs} | {spec.pdb2 for spec in pairs})
    if args.limit:
        entries = entries[: args.limit]

    rows: list[dict] = []
    for n, pdb in enumerate(entries, 1):
        try:
            found = replicate_rows(pdb, excluded[pdb])
        except Exception as exc:  # noqa: BLE001
            print(f"[{n:3d}/{len(entries)}] {pdb}: FAILED {type(exc).__name__}: {exc}")
            continue
        rows.extend(found)
        if found:
            best = ", ".join(f"{r['chain_a']}/{r['chain_b']} J={r['jaccard']:.3f}" for r in found)
            print(f"[{n:3d}/{len(entries)}] {pdb}: {best}")

    DATA.mkdir(exist_ok=True)
    out = DATA / "noise_floor.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["entry"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        jac = sorted(r["jaccard"] for r in rows)
        n = len(jac)
        print(f"\n{n} same-fold replicate chain pairs from {len({r['entry'] for r in rows})} entries")
        print(f"  jaccard: median {jac[n // 2]:.3f}  p10 {jac[n // 10]:.3f}  "
              f"p90 {jac[(9 * n) // 10]:.3f}  range {jac[0]:.3f}-{jac[-1]:.3f}")
        universe = DATA / "foldswitch_universe.jsonl"
        if universe.exists():
            records = [json.loads(line) for line in universe.open()]
            fold_jac = sorted(r["jaccard"] for r in records)
            m = len(fold_jac)
            print(f"  for contrast, the {m} fold-switch pairs: median {fold_jac[m // 2]:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
