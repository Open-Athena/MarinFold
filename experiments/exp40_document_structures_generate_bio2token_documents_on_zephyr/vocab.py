# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary + document assembly for the ``bio2token-v1`` document structure.

Document layout (the "sequence + token stream" format chosen for issue #40) —
a per-residue amino-acid sequence followed by the per-atom bio2token codes::

    <bio2token-v1> <begin_sequence> <AA_M> <AA_G> ... <begin_tokens> <bt4051> <bt3987> ... <end>

Residue tokens reuse bio2token's disambiguated names (``AA_*`` / ``RNA_*``, so
protein A and RNA A don't collide). Structure codes are ``<bt0>`` .. ``<bt4095>``
(the FSQ codebook, one per atom). The vocab is the union of the structure name,
the four markers, all residue tokens, and the 4096 code tokens.
"""

from bio2token.data.utils.tokens import AA_TO_TOKEN, RNA_TO_TOKEN

NAME = "bio2token-v1"
CODEBOOK_SIZE = 4096

STRUCTURE_TOKEN = f"<{NAME}>"
BEGIN_SEQUENCE = "<begin_sequence>"
BEGIN_TOKENS = "<begin_tokens>"
END = "<end>"
MARKERS = [STRUCTURE_TOKEN, BEGIN_SEQUENCE, BEGIN_TOKENS, END]

# Residue tokens, e.g. "<AA_M>", "<RNA_A>". Deterministic order for a stable vocab.
RESIDUE_TOKENS = [f"<{t}>" for t in AA_TO_TOKEN.values()] + [f"<{t}>" for t in RNA_TO_TOKEN.values()]
CODE_TOKENS = [f"<bt{i}>" for i in range(CODEBOOK_SIZE)]


def residue_token(one_letter: str, res_type: str) -> str:
    """Map a residue's 1-letter code + type ("aa"/"rna") to its sequence token."""
    table = AA_TO_TOKEN if res_type == "aa" else RNA_TO_TOKEN
    return f"<{table[one_letter]}>"


def all_tokens() -> list[str]:
    """The full bio2token-v1 vocabulary (markers + residues + codes)."""
    return list(MARKERS) + list(RESIDUE_TOKENS) + list(CODE_TOKENS)


def build_document(residue_tokens: list[str], atom_codes: list[int]) -> str:
    """Assemble one document string from residue tokens and per-atom codes."""
    parts = [STRUCTURE_TOKEN, BEGIN_SEQUENCE, *residue_tokens,
             BEGIN_TOKENS, *(f"<bt{c}>" for c in atom_codes), END]
    return " ".join(parts)
