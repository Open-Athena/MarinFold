# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify published documents against the coordinates they claim to describe.

Two checks the rest of the pipeline cannot make about itself.

**Reproduction.** Re-fetch a sampled system's structure and regenerate its
document; the SHA1 must match what was published. This catches a corpus built
from a different structure, a different config, or a drifted generator.

**Independent geometry.** Every inter-chain contact a document asserts is
re-measured straight from the coordinates as a minimum heavy-atom distance,
with no pyconfind involved. pyconfind's contact degree is a rotamer-based
occlusion measure, not a distance cut, so the two never have to agree
*exactly* -- but a genuine interface contact must be physically close, and a
random cross-chain pair from the same structure must not be. That contrast is
the check; the control is what makes it mean anything.

Samples are drawn by ``hash(document_id)`` so a rerun sees the same systems.

    uv run python verify_contacts.py --corpus /data/exp294_release/corpus \\
        --pinder-manifest .../pinder_selected.parquet --sample 40
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import _GENERATOR, _generator, _sql_literal  # noqa: E402
from pinder_extract import fetch_member  # noqa: E402

#: Heavy-atom separation below which two residues are touching under any
#: definition. **Not** a pass/fail bound -- see ``ROTAMER_REACH_CEILING``.
CONTACT_DISTANCE_CEILING = 8.0
#: The hard bound. contacts-v1's operator is pyconfind's contact *degree*, which
#: rebuilds side chains from the Dunbrack rotamer library and measures
#: occlusion; a residue whose side chain **can reach** its partner counts even
#: when the deposited conformer points away. Arg and Lys extend ~7 A from CA, so
#: two residues in rotamer contact can be ~14 A apart at their modelled atoms.
#: Beyond that no rotamer reaches, and the contact would be unexplainable.
ROTAMER_REACH_CEILING = 14.0
#: A contact pair must beat a random cross-chain pair by at least this factor in
#: median distance, or the geometry does not support the document's claims.
MIN_SEPARATION_FACTOR = 3.0
#: The contacts beyond ``CONTACT_DISTANCE_CEILING`` must be enriched in long,
#: flexible side chains by at least this much. This is the falsifiable part: if
#: the far tail were *not* long-side-chain enriched, the rotamer explanation
#: would be wrong and the tail would be a real defect.
MIN_LONG_SIDE_CHAIN_ENRICHMENT = 1.2
#: Side chains long enough to reach well past their own backbone.
LONG_SIDE_CHAINS = frozenset({"LYS", "ARG", "GLU", "GLN", "MET"})


def _min_heavy_atom_distance(res_a: Any, res_b: Any) -> float:
    """Closest approach of two residues, hydrogens excluded.

    Straight coordinate arithmetic -- deliberately not the contact operator the
    corpus was built with, so agreement means something.
    """
    best = float("inf")
    for atom_a in res_a:
        if atom_a.element.name == "H":
            continue
        for atom_b in res_b:
            if atom_b.element.name == "H":
                continue
            best = min(best, atom_a.pos.dist(atom_b.pos))
    return best


def _residue_index(structure: Any) -> dict[tuple[str, int], Any]:
    return {
        (chain.name, residue.seqid.num): residue
        for chain in structure[0]
        for residue in chain
    }


def verify_pinder(corpus: str, manifest: str, sample: int, seed: int) -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    # Draw from the manifest, not the shards: the shards are 19 GB and a top-N
    # over them pulls the whole corpus through memory to pick a handful of rows.
    chosen = con.execute(
        f"""
        SELECT document_id FROM read_parquet({_sql_literal(corpus + "/manifest_natural.parquet")})
        WHERE source_arm = 'pinder'
        ORDER BY hash(document_id || '{seed}')
        LIMIT {int(sample)}
        """
    ).fetchall()
    ids = ", ".join(_sql_literal(row[0]) for row in chosen)
    rows = con.execute(
        f"""
        SELECT d.document_id, d.sha1, d.contacts_emitted_inter_chain,
               m.zip_url, m.local_header_offset, m.compressed_bytes,
               m.uncompressed_bytes
        FROM read_parquet({_sql_literal(corpus + "/train/*.parquet")}) d
        JOIN read_parquet({_sql_literal(manifest)}) m
          ON m.system_id = d.document_id
        WHERE d.document_id IN ({ids})
        """
    ).fetchall()
    if len(rows) < sample:
        raise RuntimeError(f"only {len(rows)} systems available, wanted {sample}")

    gemmi, _zstd, generate = _generator()
    import random

    contact_distances: list[float] = []
    control_distances: list[float] = []
    reproduced = mismatched = 0
    inter_chain_checked = 0
    far_contacts: list[tuple[str, float]] = []
    near_residues: list[str] = []
    far_residues: list[str] = []

    for (sid, sha1, n_inter, url, lho, csize, usize) in rows:
        text = fetch_member(url, lho, csize, usize)
        structure = gemmi.read_pdb_string(text)
        structure.setup_entities()
        result = generate(structure, entry_id=sid, config=_GENERATOR["config"],
                          rotamer_library=_GENERATOR["rotamers"])
        if result is None or result.sha1 != sha1:
            mismatched += 1
            continue
        reproduced += 1
        if result.contacts_emitted_inter_chain != n_inter:
            raise RuntimeError(
                f"{sid}: regenerated {result.contacts_emitted_inter_chain} "
                f"inter-chain contacts, corpus says {n_inter}")

        index = _residue_index(structure)
        inter = [c for c in result.contacts if c.chain_i != c.chain_j]
        if len(inter) != n_inter:
            raise RuntimeError(
                f"{sid}: {len(inter)} contacts cross a chain boundary but "
                f"contacts_emitted_inter_chain is {n_inter}")
        for contact in inter:
            a = index.get((contact.chain_i, contact.resnum_i))
            b = index.get((contact.chain_j, contact.resnum_j))
            if a is None or b is None:
                raise RuntimeError(f"{sid}: contact names a residue not in the structure")
            distance = _min_heavy_atom_distance(a, b)
            contact_distances.append(distance)
            inter_chain_checked += 1
            names = [a.name, b.name]
            if distance > CONTACT_DISTANCE_CEILING:
                far_contacts.append((sid, round(distance, 2)))
                far_residues.extend(names)
            else:
                near_residues.extend(names)

        # Control: random cross-chain pairs from the same structure, drawn to
        # the same count so the two distributions are directly comparable.
        chains = [c for c in structure[0] if len(c) > 0]
        rng = random.Random(f"{sid}:{seed}")
        if len(chains) >= 2:
            for _ in range(len(inter)):
                ca, cb = rng.sample(chains, 2)
                control_distances.append(_min_heavy_atom_distance(
                    rng.choice(list(ca)), rng.choice(list(cb))))

    if not contact_distances:
        raise RuntimeError("no inter-chain contacts were checked")
    median_contact = statistics.median(contact_distances)
    median_control = statistics.median(control_distances)
    factor = median_control / median_contact
    within = sum(d <= CONTACT_DISTANCE_CEILING for d in contact_distances)
    # pyconfind measures rotamer occlusion, not a distance cut: a residue whose
    # side chain can *reach* its partner counts, even when the deposited
    # conformer points away. If the far tail is that effect, it is enriched in
    # the long, flexible side chains -- so measure the enrichment rather than
    # assume it.
    from collections import Counter
    near_counts, far_counts = Counter(near_residues), Counter(far_residues)
    near_long = sum(near_counts[r] for r in LONG_SIDE_CHAINS) / max(1, sum(near_counts.values()))
    far_long = sum(far_counts[r] for r in LONG_SIDE_CHAINS) / max(1, sum(far_counts.values()))
    enrichment = (far_long / near_long) if near_long else None
    report = {
        "systems_sampled": len(rows),
        "long_side_chain_fraction_within_ceiling": round(near_long, 4),
        "long_side_chain_fraction_beyond_ceiling": round(far_long, 4),
        "long_side_chain_enrichment_beyond_ceiling": (
            round(enrichment, 2) if enrichment is not None else None),
        "most_common_residues_beyond_ceiling": far_counts.most_common(6),
        "documents_reproduced_byte_identical": reproduced,
        "documents_mismatched": mismatched,
        "inter_chain_contacts_checked": inter_chain_checked,
        "contact_distance_median_angstrom": round(median_contact, 3),
        "contact_distance_p95_angstrom": round(
            sorted(contact_distances)[int(0.95 * len(contact_distances))], 3),
        "contact_distance_max_angstrom": round(max(contact_distances), 3),
        "fraction_within_ceiling": round(within / len(contact_distances), 6),
        "contact_distance_ceiling_angstrom": CONTACT_DISTANCE_CEILING,
        "rotamer_reach_ceiling_angstrom": ROTAMER_REACH_CEILING,
        "control_distance_median_angstrom": round(median_control, 3),
        "separation_factor": round(factor, 2),
        "contacts_beyond_ceiling": far_contacts[:10],
    }
    problems = []
    if mismatched:
        problems.append(f"{mismatched} documents did not reproduce byte-identically")
    unreachable = [d for d in contact_distances if d > ROTAMER_REACH_CEILING]
    if unreachable:
        problems.append(
            f"{len(unreachable)} contacts exceed {ROTAMER_REACH_CEILING} A, which "
            "no rotamer can bridge")
    if factor < MIN_SEPARATION_FACTOR:
        problems.append(
            f"contacts are only {factor:.2f}x closer than random cross-chain "
            f"pairs, expected >= {MIN_SEPARATION_FACTOR}")
    if within != len(contact_distances) and (
            enrichment is None or enrichment < MIN_LONG_SIDE_CHAIN_ENRICHMENT):
        problems.append(
            f"the {len(contact_distances) - within} contacts beyond "
            f"{CONTACT_DISTANCE_CEILING} A are not long-side-chain enriched "
            f"(x{enrichment}), so the rotamer-reach explanation does not hold")
    report["problems"] = problems
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--pinder-manifest", required=True)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=294)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = verify_pinder(args.corpus, args.pinder_manifest, args.sample, args.seed)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
