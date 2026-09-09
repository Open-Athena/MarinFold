# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does pyconfind need input side chains, or does it rebuild them from rotamers?

The premise of the whole experiment. If contact degrees match between an
all-atom structure and the same structure stripped to backbone atoms, then a
ProteinMPNN redesign (original backbone + new residue names) is a legal
pyconfind input and the redesigned corpus is computed under *exactly* the
same contact operator as `contacts_v1`.

Pinned as a regression test in ``tests/test_backbone.py``; this script is the
readable version that prints the table quoted in the README.

Run (workstation, against the local PDB mirror)::

    PYTHONPATH=../../marinfold uv run --no-project --with gemmi --with pyconfind \
        python probe_pyconfind.py /data/tim/af3-db/mmcif_files/{1crn,1ubq,101m}.cif
"""

import sys

import gemmi

from backbone import BACKBONE_ATOMS, prepare_structure, strip_to_backbone
from marinfold.document_structures.contacts_v1.parse import analyze_structure


def degrees(structure) -> dict[tuple[int, int], float]:
    analysis = analyze_structure(structure, entry_id="probe")
    return {(c.seq_i, c.seq_j): c.degree for c in analysis.contacts}


def main(paths: list[str]) -> None:
    print(f"backbone atoms kept: {sorted(BACKBONE_ATOMS)}")
    print(f"{'structure':>12} {'residues':>9} {'all-atom':>9} {'backbone':>9} "
          f"{'identical':>10} {'max|delta|':>11}")
    for path in paths:
        structure = prepare_structure(gemmi.read_structure(path))
        try:
            full = degrees(structure)
            stripped = degrees(strip_to_backbone(structure))
        except ValueError as exc:
            print(f"{path.split('/')[-1]:>12}  skipped: {exc}")
            continue
        n_res = sum(1 for chain in structure[0] for _ in chain)
        shared = set(full) & set(stripped)
        worst = max((abs(full[k] - stripped[k]) for k in shared), default=0.0)
        print(f"{path.split('/')[-1]:>12} {n_res:9d} {len(full):9d} {len(stripped):9d} "
              f"{str(set(full) == set(stripped)):>10} {worst:11.3e}")


if __name__ == "__main__":
    main(sys.argv[1:])
