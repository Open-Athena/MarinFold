# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end smoke: PDB mirror -> backbone -> ProteinMPNN -> contacts-v1.

Runs the whole exp266 per-row path on a workstation GPU, on real structures,
before anything touches a cluster. Prints the three things the pilot must
measure: throughput, ProteinMPNN composition drift, and contact-density drift.

    uv run --no-project --with proteinmpnn --with 'numpy<2' --with gemmi \
        --with torch python smoke_local.py --limit 32
"""

from __future__ import annotations

import argparse
import collections
import random
import time
from pathlib import Path

import gemmi

from backbone import (
    backbone_coords,
    prepare_structure,
    relabel_sequence,
    residue_sequence,
    strip_to_backbone,
)
from redesign import BackboneEntry, DESIGN_TEMPERATURES, batch_by_exact_length, design_batch
from marinfold.document_structures.contacts_v1 import generate_document

MIRROR = Path("/data/tim/af3-db/mmcif_files")


def load_backbones(limit: int, seed: int = 0) -> list[tuple[BackboneEntry, gemmi.Structure]]:
    stems = sorted(p.stem for p in MIRROR.glob("*.cif"))
    random.Random(seed).shuffle(stems)
    out = []
    for stem in stems:
        if len(out) >= limit:
            break
        try:
            st = prepare_structure(gemmi.read_structure(str(MIRROR / f"{stem}.cif")))
            if len(st) == 0 or sum(1 for _ in st[0]) != 1:
                continue                      # monomers only, as contacts-v1 requires
            bb = strip_to_backbone(st)
            seq = residue_sequence(bb)
            if not (30 <= len(seq) <= 800) or "X" in seq:
                continue                      # designed-in filter predicates
            _chains, coords = backbone_coords(bb)
        except (ValueError, RuntimeError):
            continue
        out.append((BackboneEntry(stem, seq, coords), bb))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-batch-residues", type=int, default=16000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    pairs = load_backbones(args.limit)
    entries = [e for e, _ in pairs]
    structures = {e.entry_id: bb for e, bb in pairs}
    t_load = time.perf_counter() - t0
    total_res = sum(e.length for e in entries)
    print(f"loaded {len(entries)} monomers ({total_res} residues) in {t_load:.1f}s")

    t0 = time.perf_counter()
    designs = []
    for batch in batch_by_exact_length(
        entries, max_batch=args.max_batch, max_batch_residues=args.max_batch_residues
    ):
        designs.extend(design_batch(batch, device=args.device))
    t_mpnn = time.perf_counter() - t0
    n_seq = len(designs)
    print(f"ProteinMPNN: {n_seq} sequences in {t_mpnn:.1f}s "
          f"= {1000 * t_mpnn / n_seq:.1f} ms/sequence "
          f"({t_mpnn / len(entries) * 1000:.0f} ms/backbone for {len(DESIGN_TEMPERATURES)} designs)")

    by_temp = collections.defaultdict(list)
    for d in designs:
        by_temp[d.mpnn_temperature].append(d)
    print("\n  T     n   identity_to_native   mpnn_score")
    for temp in sorted(by_temp):
        ds = by_temp[temp]
        ident = sum(d.identity_to_native for d in ds) / len(ds)
        score = sum(d.mpnn_score for d in ds) / len(ds)
        print(f" {temp:4.1f} {len(ds):5d}   {ident:.3f}                {score:.3f}")

    # Composition drift — the risk flagged in the issue's success criteria.
    native_counts = collections.Counter("".join(e.native_sequence for e in entries))
    design_counts = collections.Counter("".join(d.sequence for d in designs))
    n_nat = sum(native_counts.values())
    n_des = sum(design_counts.values())
    print("\n  AA   native%   design%    delta")
    for aa in sorted(set(native_counts) | set(design_counts),
                     key=lambda a: -design_counts[a] / n_des):
        nat = 100 * native_counts[aa] / n_nat
        des = 100 * design_counts[aa] / n_des
        flag = "  <<<" if abs(des - nat) > 3.0 else ""
        print(f"   {aa}   {nat:6.2f}   {des:6.2f}   {des - nat:+6.2f}{flag}")

    # Contact-density drift: do redesigned documents assert fewer contacts?
    t0 = time.perf_counter()
    nat_density, des_density, n_docs = [], [], 0
    for entry in entries:
        bb = structures[entry.entry_id]
        native_doc = generate_document(
            relabel_sequence(bb, entry.native_sequence), entry_id=entry.entry_id
        )
        if native_doc is None:
            continue
        nat_density.append(native_doc.contacts_emitted / entry.length)
        for d in (x for x in designs if x.entry_id == entry.entry_id):
            doc = generate_document(
                relabel_sequence(bb, d.sequence), entry_id=entry.entry_id
            )
            if doc is not None:
                des_density.append(doc.contacts_emitted / entry.length)
                n_docs += 1
    t_doc = time.perf_counter() - t0
    print(f"\ncontacts-v1: {n_docs} redesigned documents in {t_doc:.1f}s "
          f"= {1000 * t_doc / max(n_docs, 1):.0f} ms/document (CPU, 1 core)")
    if nat_density and des_density:
        nat_m = sum(nat_density) / len(nat_density)
        des_m = sum(des_density) / len(des_density)
        print(f"contacts per residue: native {nat_m:.3f}  designed {des_m:.3f}  "
              f"ratio {des_m / nat_m:.3f}")


if __name__ == "__main__":
    main()
