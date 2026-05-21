# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the canonical 1..N residue sequence from a Protenix GT CIF.

The Protenix GT mmCIFs (one per FoldBench monomer, RCSB biological
assembly) have a single polypeptide(L) entity whose ``full_sequence``
gives the 1..N canonical sequence — *including* unresolved residues.
That's the same N that exp12's ``_read_protein_coords_from_cif`` uses
for scoring, so building the 1B prompt from this list keeps the
distogram index space aligned with the GT distance matrix.

Non-canonical residues (e.g. MSE selenomethionine, SEP phospho-serine)
are mapped to ``UNK`` to match the v1 tokenizer's residue vocab.
"""

from dataclasses import dataclass
from pathlib import Path

import gemmi

from vocab import AMINO_ACIDS  # exp1 path dep — same canonical 20


# Canonical 20 amino-acid set. Non-canonical residues at the
# entity_poly_seq level get UNK in the prompt; this matches
# exp1.parse._CANONICAL_20.
_CANONICAL_20 = frozenset(AMINO_ACIDS)


@dataclass(frozen=True)
class CanonicalSequence:
    """The canonical 1..N residue list extracted from a Protenix GT CIF.

    ``residue_names`` is length-N, 3-letter (e.g. ``ALA``), with
    non-canonical residues mapped to ``UNK``. The 1-indexed position
    matches gemmi's ``label_seq_id`` and so matches exp12's CB-CB
    distance matrix indexing exactly.
    """

    n_residues: int
    residue_names: tuple[str, ...]


def read_canonical_sequence(cif_path: Path) -> CanonicalSequence:
    """Pull the single polypeptide(L) entity's 1..N sequence.

    Raises:
        ValueError: when the CIF has zero or >1 L-peptide entities
            (FoldBench monomers always have exactly one; failing
            loudly here matches exp12's invariant).
    """
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()

    peptide_entities = [
        e for e in structure.entities
        if e.entity_type == gemmi.EntityType.Polymer
        and e.polymer_type == gemmi.PolymerType.PeptideL
    ]
    if not peptide_entities:
        raise ValueError(f"No polypeptide(L) entity in {cif_path}")
    if len(peptide_entities) > 1:
        raise ValueError(
            f"{cif_path} has {len(peptide_entities)} polypeptide(L) entities; "
            f"exp20 only handles monomers."
        )

    entity = peptide_entities[0]
    seq: list[str] = []
    for raw_name in entity.full_sequence:
        # entity.full_sequence is a list of strings like 'ALA' or
        # comma-separated alternatives like 'ASP,ASN' for ambiguous
        # entries; pick the first.
        name = str(raw_name).split(",")[0].strip().upper()
        seq.append(name if name in _CANONICAL_20 else "UNK")

    return CanonicalSequence(n_residues=len(seq), residue_names=tuple(seq))
