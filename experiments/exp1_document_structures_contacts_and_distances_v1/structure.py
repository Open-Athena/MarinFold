# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 DocumentStructure.

The canonical MarinFold protein document format. A document looks
like::

    <contacts-and-distances-v1>
    <begin_sequence> <AA_1> ... <AA_n>
    <begin_statements>
    <long-range-contact> <p_i> <p_j>
    <medium-range-contact> <p_i> <p_j>
    <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>
    <short-range-contact> <p_i> <p_j>
    <plddt_80_85>
    <end>

See ``README.md`` and the `HF dataset page
<https://huggingface.co/datasets/timodonnell/protein-docs>`_ for the
full format spec.

This file is intentionally one module — vocab, parsing, generation,
and evaluation are all here. We can split later if it grows past
~1000 lines, but co-location keeps the format definition reviewable
in one place.

NOTE: this is the **step-1 skeleton** — vocab + ``tokens()`` are
final and tested. ``iter_inputs`` / ``iter_ground_truth`` /
``generate_documents`` / ``evaluate`` raise ``NotImplementedError``
and will be filled in in subsequent commits.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from marinfold_document_structures import DocumentStructure, EvalResult


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
#
# Order is load-bearing. Token IDs derived from this list must stay
# stable for every checkpoint trained against the v1 vocab. Append-
# only — never reorder. (Reordering is a v2 event: new structure,
# new tokenizer, new experiment.)
#
# Source of truth ported from
# experiments/exp0_models_protein_docs_initial_port/create_protein_tokenizer.py
# and timodonnell/LlamaFold-experiments/.../exp6_contact_prediction/src/data.py.

CONTROL_TOKENS = [
    "<contacts-and-distances-v1>",
    "<begin_sequence>",
    "<begin_statements>",
    "<end>",
]

CONTACT_TYPES = [
    # CASP-standard separation ranges. Defined by CB-CB <= 8 Å.
    "<long-range-contact>",     # sequence separation >= 24
    "<medium-range-contact>",   # 12 .. 24
    "<short-range-contact>",    # 6 .. 12
]

DISTANCE_MARKER = ["<distance>"]

# 64 bins at 0.5 Å resolution: <d0.5>, <d1.0>, ..., <d32.0>.
DISTANCE_BINS = [f"<d{i * 0.5:.1f}>" for i in range(1, 65)]

PLDDT_BINS = [
    "<plddt_lt70>",
    "<plddt_70_75>",
    "<plddt_75_80>",
    "<plddt_80_85>",
    "<plddt_85_90>",
    "<plddt_90_95>",
    "<plddt_95_100>",
]

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]

ATOM_NAMES = [
    # Order is load-bearing: matches
    # exp0_models_protein_docs_initial_port/create_protein_tokenizer.py
    # and LlamaFold-experiments/.../exp6_contact_prediction/src/data.py.
    # Reordering would change token IDs and break every existing v1
    # checkpoint. Backbone atoms (N, CA, C, O, OXT) are interleaved
    # alphabetically — do NOT group them.
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
    "SD", "SG", "OXT",
]

# Position tokens <p0> through <p2700>. The upper bound caps the
# longest chain we tokenize; longer chains are dropped at generation
# time (PROBABLY shouldn't appear in training data anyway).
MAX_POSITION = 2700
POSITION_TOKENS = [f"<p{i}>" for i in range(MAX_POSITION + 1)]

UNK_TOKEN = ["<UNK>"]


# Backbone-set used by the standard residue → valid atoms lookup
# below. Used at generation time to discard atom records that the
# vocab doesn't have a token for.
_BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Per-residue valid atom set. Atom records outside this set get
# dropped during doc generation (they're either nonstandard or
# alt-loc artifacts the format doesn't represent).
VALID_ATOMS: dict[str, set[str]] = {
    "ALA": _BACKBONE | {"CB"},
    "ARG": _BACKBONE | {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": _BACKBONE | {"CB", "CG", "OD1", "ND2"},
    "ASP": _BACKBONE | {"CB", "CG", "OD1", "OD2"},
    "CYS": _BACKBONE | {"CB", "SG"},
    "GLN": _BACKBONE | {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": _BACKBONE | {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": _BACKBONE,
    "HIS": _BACKBONE | {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": _BACKBONE | {"CB", "CG1", "CG2", "CD1"},
    "LEU": _BACKBONE | {"CB", "CG", "CD1", "CD2"},
    "LYS": _BACKBONE | {"CB", "CG", "CD", "CE", "NZ"},
    "MET": _BACKBONE | {"CB", "CG", "SD", "CE"},
    "PHE": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": _BACKBONE | {"CB", "CG", "CD"},
    "SER": _BACKBONE | {"CB", "OG"},
    "THR": _BACKBONE | {"CB", "OG1", "CG2"},
    "TRP": _BACKBONE | {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": _BACKBONE | {"CB", "CG1", "CG2"},
}


def _all_domain_tokens() -> list[str]:
    """Canonical 2838-token domain vocabulary in deterministic order.

    The published tokenizer prepends ``<pad>`` and ``<eos>`` (specials
    at ids 0 and 1), giving 2840 total. ``build_tokenizer(structure)``
    handles the prepend; this function returns the domain tokens
    only.

    The category-by-category order here (control → contact-types →
    distance-marker → distance-bins → plddt-bins → AAs → atoms →
    positions → UNK) and the within-category order are both
    load-bearing. Any reordering breaks compatibility with every
    checkpoint trained against the v1 vocab.
    """
    out: list[str] = []
    out += CONTROL_TOKENS
    out += CONTACT_TYPES
    out += DISTANCE_MARKER
    out += DISTANCE_BINS
    out += PLDDT_BINS
    out += [f"<{aa}>" for aa in AMINO_ACIDS]
    out += [f"<{atom}>" for atom in ATOM_NAMES]
    out += POSITION_TOKENS
    out += UNK_TOKEN
    return out


# --------------------------------------------------------------------------
# DocumentStructure implementation
# --------------------------------------------------------------------------


class ContactsAndDistancesV1:
    """The contacts-and-distances-v1 document structure.

    Stateless apart from the cached token list — safe to construct
    eagerly via :func:`get_structure`.
    """

    name = "contacts-and-distances-v1"
    context_length = 8192

    def __init__(self) -> None:
        self._tokens = _all_domain_tokens()

    def tokens(self) -> list[str]:
        # Return a copy so callers can't accidentally mutate the
        # canonical list (which would corrupt every subsequent
        # build_tokenizer call).
        return list(self._tokens)

    # ---- generate side --------------------------------------------------

    def iter_inputs(self, path: Path) -> Iterator[Any]:
        raise NotImplementedError(
            "iter_inputs is not yet implemented — coming in the next commit "
            "(PDB + mmCIF parsing via gemmi). See the experiment README's "
            "Approach section."
        )

    def generate_documents(
        self,
        input_records: Iterator[Any],
        *,
        context_length: int | None = None,
        num_docs: int | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError(
            "generate_documents is not yet implemented — coming in the "
            "next commit."
        )

    # ---- evaluate side --------------------------------------------------

    def iter_ground_truth(self, path: Path) -> Iterator[Any]:
        raise NotImplementedError(
            "iter_ground_truth is not yet implemented — coming in the next "
            "commit (shares the same parser as iter_inputs)."
        )

    def evaluate(
        self,
        *,
        model_path: str,
        ground_truth_records: Iterator[Any],
    ) -> EvalResult:
        raise NotImplementedError(
            "evaluate is not yet implemented — coming in a subsequent commit "
            "(vllm-backed rollout eval)."
        )


def get_structure() -> DocumentStructure:
    """Entry point read by the marinfold-document-structure CLI."""
    return ContactsAndDistancesV1()
