# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical vocabulary for the contacts-and-distances-v1 format.

Private to this experiment dir. Imported by ``generate.py`` and
``inference.py`` — both must expose the same ordered token list via
``tokens()`` (the ``Generator`` / ``Inference`` Protocols enforce
this implicitly: any mismatch means the tokenizers they build are
incompatible).

Order is **load-bearing**. Token IDs derived from this list must
stay stable for every checkpoint trained against the v1 vocab.
Append-only — never reorder. (Reordering is a v2 event: new
structure, new tokenizer, new experiment.)

Source-of-truth ported from
``exp0_models_protein_docs_initial_port/create_protein_tokenizer.py``
and ``timodonnell/LlamaFold-experiments/.../exp6_contact_prediction/src/data.py``.
"""


NAME = "contacts-and-distances-v1"
CONTEXT_LENGTH = 8192


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
    # Backbone atoms (N, CA, C, O, OXT) are interleaved alphabetically
    # with sidechain atoms — do NOT regroup; that changes IDs and
    # breaks every v1 checkpoint.
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
    "SD", "SG", "OXT",
]

# Position tokens <p0> through <p2700>. Caps the longest chain we
# tokenize; longer chains are dropped at generation time.
MAX_POSITION = 2700
POSITION_TOKENS = [f"<p{i}>" for i in range(MAX_POSITION + 1)]

UNK_TOKEN = ["<UNK>"]


def all_domain_tokens() -> list[str]:
    """Return the canonical 2838-token domain vocabulary in deterministic order.

    The published tokenizer prepends ``<pad>`` and ``<eos>`` (specials
    at ids 0 and 1), giving 2840 total. ``build_tokenizer`` handles
    the prepend; this function returns the domain tokens only.

    The category-by-category order here (control → contact-types →
    distance-marker → distance-bins → plddt-bins → AAs → atoms →
    positions → UNK) and the within-category order are both
    load-bearing.
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
