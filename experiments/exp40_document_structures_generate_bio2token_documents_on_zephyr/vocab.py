# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary + document assembly for the ``bio2token-v2`` document structure.

A **self-describing** layout: instead of a flat stream of per-atom codes in a
fixed canonical order (v1, which forced the model to index into the stream to
know which atom each code was), every atom is emitted as a triple
``<pN> <atom-name> <btC>`` — residue position, atom name, bio2token code — and
the triples are shuffled into a random order (per document, seeded by the entry
id). Each token therefore says which atom it belongs to, so order carries no
information and the model never has to count::

    <bio2token-v2>
      <begin_sequence> <p0> <MET> <p1> <ALA> ...          # residues, in order
      <begin_statements> <p5> <CA> <bt508> <p3> <CE> <bt502> ...   # atoms, shuffled
    <end>

Token strings are **reused verbatim from the contacts document structures**
(``marinfold/.../contacts_and_distances_v1/vocab.py``) so bio2token documents
share a token space with them and can be mixed under one tokenizer: three-letter
amino acids (``<MET>`` …), atom names (``<CA>`` …), position indices (``<pN>``),
``<UNK>``, and the section markers. Only the document-type marker
(``<bio2token-v2>``) and the 4096 bio2token codes (``<bt0>`` … ``<bt4095>``) are
minted here — nothing in the contacts space encodes a learned atom token.
"""

import hashlib
import random

NAME = "bio2token-v2"
CODEBOOK_SIZE = 4096  # prod(FSQ levels) = 4^6

# --- Document-type marker + section markers ---------------------------------
# The doc-type marker is minted here; the section markers + <end> are the
# shared strings from the contacts structures.
STRUCTURE_TOKEN = f"<{NAME}>"
BEGIN_SEQUENCE = "<begin_sequence>"
BEGIN_STATEMENTS = "<begin_statements>"
END = "<end>"
MARKERS = [STRUCTURE_TOKEN, BEGIN_SEQUENCE, BEGIN_STATEMENTS, END]

# --- Shared strings, copied verbatim from contacts-and-distances-v1 ----------
# Source of truth: marinfold/.../contacts_and_distances_v1/vocab.py. Kept as
# literal copies (not an import) so this experiment stays self-contained; if the
# canonical lists change, mirror them here. Order is irrelevant to a document
# (which emits strings), but matches upstream for easy diffing.
AMINO_ACIDS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
)
ATOM_NAMES = (
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
    "SD", "SG", "OXT",
)
UNK = "<UNK>"
MAX_POSITION = 2700  # <p0> .. <p2700>; longer chains are dropped at generation

_AMINO_ACID_SET = frozenset(AMINO_ACIDS)
_ATOM_NAME_SET = frozenset(ATOM_NAMES)

# --- bio2token codes (minted here — no shared analog) ------------------------
CODE_TOKENS = [f"<bt{i}>" for i in range(CODEBOOK_SIZE)]


def position_token(index: int) -> str:
    """Residue position token ``<pN>`` (reused from the contacts structures)."""
    return f"<p{index}>"


def residue_token(res_name: str) -> str:
    """Three-letter residue → ``<RES>``; non-standard residues → ``<UNK>``.

    AlphaFold-DB inputs are the standard 20 amino acids; the ``<UNK>`` fallback
    only fires for non-standard residues (selenocysteine, ASX/GLX/XAA), which
    do not occur in that corpus.
    """
    res_name = res_name.upper()
    return f"<{res_name}>" if res_name in _AMINO_ACID_SET else UNK


def atom_token(atom_name: str) -> str:
    """Atom name → ``<NAME>``; names outside the shared set → ``<UNK>``.

    The only bio2token atom name absent from the shared set is ``SE``
    (selenocysteine), which does not occur in the AlphaFold-DB corpus.
    """
    return f"<{atom_name}>" if atom_name in _ATOM_NAME_SET else UNK


def code_token(code: int) -> str:
    return f"<bt{code}>"


def _seed_from(entry_id: str) -> int:
    """Deterministic per-document seed so a document's atom shuffle is stable
    across runs (Python's ``hash`` is salted per process, so use SHA-256)."""
    return int.from_bytes(hashlib.sha256(entry_id.encode()).digest()[:8], "big")


def build_document(
    sequence: list[tuple[int, str]],
    atoms: list[tuple[int, str, int]],
    *,
    entry_id: str,
) -> str:
    """Assemble one ``bio2token-v2`` document.

    ``sequence`` is ``(position, res_name)`` per residue in chain order;
    ``atoms`` is ``(position, atom_name, code)`` per kept atom in canonical
    order. The atom triples are shuffled with a per-document seed derived from
    ``entry_id`` — the whole point of the format, so order carries no signal.
    """
    parts = [STRUCTURE_TOKEN, BEGIN_SEQUENCE]
    for pos, res_name in sequence:
        parts += [position_token(pos), residue_token(res_name)]

    shuffled = list(atoms)
    random.Random(_seed_from(entry_id)).shuffle(shuffled)
    parts.append(BEGIN_STATEMENTS)
    for pos, atom_name, code in shuffled:
        parts += [position_token(pos), atom_token(atom_name), code_token(code)]

    parts.append(END)
    return " ".join(parts)


def all_tokens() -> list[str]:
    """The full bio2token-v2 vocabulary (markers + residues + atoms + positions
    + UNK + codes). Reused strings and minted strings are disjoint."""
    residues = [f"<{aa}>" for aa in AMINO_ACIDS]
    atoms = [f"<{a}>" for a in ATOM_NAMES]
    positions = [position_token(i) for i in range(MAX_POSITION + 1)]
    return list(MARKERS) + residues + atoms + positions + [UNK] + list(CODE_TOKENS)
