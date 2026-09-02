# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end smoke over the real staged path.

PDB mirror -> strip -> **encode/decode as a staged backbone row** -> ProteinMPNN
-> contacts-v1. The encode/decode hop is deliberately in the loop: it is what
the CoreWeave workers actually consume, and a smoke that skipped it would not
be testing the pipeline that runs.

Prints the three things the pilot must measure: throughput, ProteinMPNN
composition drift, and contact-density drift.

Run (the workstation's driver is too old for the default torch wheel, so pin
the cu121 build)::

    PYTHONPATH=../../marinfold uv run --no-project --python 3.11 \\
        --index https://download.pytorch.org/whl/cu121 \\
        --index-strategy unsafe-best-match \\
        --with torch==2.4.1 --with proteinmpnn --with 'numpy<2' \\
        --with gemmi --with pyconfind --with fsspec \\
        python smoke_local.py --limit 48
"""

from __future__ import annotations

import argparse
import collections
import random
import time
from pathlib import Path

import gemmi

from backbone import (
    backbone_coords_from_row,
    encode_backbone,
    prepare_structure,
    strip_to_backbone,
)
from generate_rows import _load_rotamer_library, documents_for_designs
from redesign import (
    DESIGN_TEMPERATURES,
    BackboneEntry,
    batch_by_exact_length,
    design_batch,
)

MIRROR = Path("/data/tim/af3-db/mmcif_files")


def load_staged_rows(limit: int, seed: int = 0) -> list[dict]:
    """Mimic Stage A2: mmCIF -> staged backbone row, applying the same filters."""
    stems = sorted(p.stem for p in MIRROR.glob("*.cif"))
    random.Random(seed).shuffle(stems)
    rows: list[dict] = []
    for stem in stems:
        if len(rows) >= limit:
            break
        try:
            st = prepare_structure(gemmi.read_structure(str(MIRROR / f"{stem}.cif")))
            if len(st) == 0 or sum(1 for _ in st[0]) != 1:
                continue                       # monomers only, as contacts-v1 requires
            backbone = strip_to_backbone(st)
            row = encode_backbone(backbone) | {"entry_id": stem}
        except (ValueError, RuntimeError):
            continue                           # designed-in filters; see stage_rows
        if not (30 <= len(row["sequence"]) <= 800):
            continue
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-batch-residues", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rows = load_staged_rows(args.limit)
    by_id = {r["entry_id"]: r for r in rows}
    total_res = sum(len(r["sequence"]) for r in rows)
    staged_bytes = sum(len(r["coords_milli"]) * 4 + len(r["ca_plddt"]) * 4
                       + len(r["sequence"]) for r in rows)
    print(f"staged {len(rows)} monomers ({total_res} residues) in "
          f"{time.perf_counter() - t0:.1f}s — {staged_bytes / len(rows) / 1024:.1f} KB/protein "
          f"uncompressed, so ~{staged_bytes / len(rows) * 3_963_003 / 1e9:.0f} GB "
          f"for the full corpus")

    entries = [
        BackboneEntry(r["entry_id"], r["sequence"], backbone_coords_from_row(r))
        for r in rows
    ]

    t0 = time.perf_counter()
    designs = []
    for batch in batch_by_exact_length(
        entries, max_batch=args.max_batch, max_batch_residues=args.max_batch_residues
    ):
        designs.extend(design_batch(batch, device=args.device))
    t_mpnn = time.perf_counter() - t0
    print(f"ProteinMPNN: {len(designs)} sequences in {t_mpnn:.1f}s "
          f"= {1000 * t_mpnn / len(designs):.1f} ms/sequence "
          f"({1000 * t_mpnn / len(rows):.0f} ms/backbone for "
          f"{len(DESIGN_TEMPERATURES)} designs)")

    by_temp = collections.defaultdict(list)
    for d in designs:
        by_temp[d.mpnn_temperature].append(d)
    print("\n  T     n   identity_to_native   mpnn_score")
    for temp in sorted(by_temp):
        ds = by_temp[temp]
        print(f" {temp:4.1f} {len(ds):5d}   "
              f"{sum(d.identity_to_native for d in ds) / len(ds):.3f}                "
              f"{sum(d.mpnn_score for d in ds) / len(ds):.3f}")

    # Composition drift — the risk flagged in the issue's success criteria.
    native_counts = collections.Counter("".join(r["sequence"] for r in rows))
    design_counts = collections.Counter("".join(d.sequence for d in designs))
    n_nat, n_des = sum(native_counts.values()), sum(design_counts.values())
    print("\n  AA   native%   design%    delta")
    for aa in sorted(set(native_counts) | set(design_counts),
                     key=lambda a: -design_counts[a] / n_des):
        nat, des = 100 * native_counts[aa] / n_nat, 100 * design_counts[aa] / n_des
        flag = "  <<<" if abs(des - nat) > 3.0 else ""
        print(f"   {aa}   {nat:6.2f}   {des:6.2f}   {des - nat:+6.2f}{flag}")

    # Contact-density drift: do redesigned documents assert fewer contacts?
    rotamers = _load_rotamer_library()
    designs_by_entry: dict[str, list] = collections.defaultdict(list)
    for d in designs:
        designs_by_entry[d.entry_id].append(d)

    t0 = time.perf_counter()
    nat_density, des_density, n_docs = [], [], 0
    for entry_id, row in by_id.items():
        length = len(row["sequence"])
        native = documents_for_designs(
            row, [_NativeDesign(entry_id, row["sequence"])], rotamer_library=rotamers)
        if not native:
            continue
        nat_density.append(native[0]["contacts_emitted"] / length)
        for record in documents_for_designs(
            row, designs_by_entry[entry_id], rotamer_library=rotamers
        ):
            des_density.append(record["contacts_emitted"] / length)
            n_docs += 1
    t_doc = time.perf_counter() - t0
    print(f"\ncontacts-v1: {n_docs} redesigned documents in {t_doc:.1f}s "
          f"= {1000 * t_doc / max(n_docs, 1):.0f} ms/document (CPU, 1 core)")
    if nat_density and des_density:
        nat_m = sum(nat_density) / len(nat_density)
        des_m = sum(des_density) / len(des_density)
        print(f"contacts per residue: native {nat_m:.3f}  designed {des_m:.3f}  "
              f"ratio {des_m / nat_m:.3f}")


class _NativeDesign:
    """The native sequence dressed as a Design, so the reference document goes
    through exactly the same code path as the redesigned ones."""

    design_index = -1
    mpnn_temperature = 0.0
    mpnn_score = 0.0
    identity_to_native = 1.0

    def __init__(self, entry_id: str, sequence: str) -> None:
        self.entry_id = entry_id
        self.sequence = sequence


if __name__ == "__main__":
    main()
