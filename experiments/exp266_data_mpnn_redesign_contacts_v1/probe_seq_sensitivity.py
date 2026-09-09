# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""On a FIXED backbone, how much does the contacts-v1 contact set move when
only the sequence changes?

Bounds what 8 ProteinMPNN redesigns per protein can possibly teach. Near
sequence-invariance would mean the 8 designs are 8 near-duplicate documents;
strong sequence-dependence means each design is a genuinely different label —
and that the corpus currently shows every fold with exactly one sequence.

Output: ``data/sequence_sensitivity_probe.csv``.

Run (workstation, against the local PDB mirror)::

    PYTHONPATH=../../marinfold uv run --no-project --with gemmi --with pyconfind \
        python probe_seq_sensitivity.py /data/tim/af3-db/mmcif_files/{1crn,1ubq,101m}.cif
"""

import csv
import random
import sys
from pathlib import Path

import gemmi

from backbone import AA1_TO_AA3, prepare_structure, relabel_sequence, residue_sequence, strip_to_backbone
from marinfold.document_structures.contacts_v1 import GenerationConfig
from marinfold.document_structures.contacts_v1.parse import analyze_structure

CONFIG = GenerationConfig()
OUT = Path("data/sequence_sensitivity_probe.csv")


def selected(structure) -> set[tuple[int, int]]:
    """The contacts a contacts-v1 document would actually assert."""
    analysis = analyze_structure(structure, entry_id="probe")
    return {
        (c.seq_i, c.seq_j)
        for c in analysis.contacts
        if c.degree >= CONFIG.min_contact_degree
        and abs(c.seq_i - c.seq_j) >= CONFIG.min_seq_separation
    }


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main(paths: list[str]) -> None:
    rng = random.Random(0)
    alphabet = sorted(AA1_TO_AA3)
    rows = []
    for path in paths:
        backbone = strip_to_backbone(prepare_structure(gemmi.read_structure(path)))
        native = residue_sequence(backbone)
        if "X" in native:
            print(f"{path}: skipped (non-canonical residues)")
            continue
        base = selected(backbone)
        stem = Path(path).stem
        print(f"\n{stem}  {len(native)} res  {len(base)} selected contacts (native)")

        variants = {
            "shuffled_native": "".join(rng.sample(native, len(native))),
            "random_uniform": "".join(rng.choice(alphabet) for _ in native),
            "poly_LEU": "L" * len(native),
            "poly_ALA": "A" * len(native),
            "poly_GLY": "G" * len(native),
        }
        for label, sequence in variants.items():
            variant = selected(relabel_sequence(backbone, sequence))
            j = jaccard(base, variant)
            recall = len(base & variant) / max(len(base), 1)
            print(f"   {label:16s} n={len(variant):5d}  Jaccard={j:.3f}  recall={recall:.3f}")
            rows.append({
                "structure": stem, "residues": len(native),
                "native_contacts": len(base), "variant": label,
                "variant_contacts": len(variant),
                "jaccard_vs_native": round(j, 3),
                "recall_of_native": round(recall, 3),
            })

    if rows:
        OUT.parent.mkdir(exist_ok=True)
        with OUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
