# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Invert a contacts-v1 document back into (L, sequence, ground-truth contacts).

Vendored from exp98's ``select_targets.py`` (#98), which exp163 in turn imported
by ``sys.path`` hack.  Copied here instead so exp230 has no cross-experiment
import: the function is 20 lines, and the alternative — #216's
``contacts_v1.read.sequence_from_document`` — is still on an unmerged branch.

The inversion is exact because a contacts-v1 document carries everything it
needs: ``<n-term> <pN>`` pins the reading frame, ``<pN> <XXX>`` pairs give the
residue at every ring index, and the ring is walked modulo 2000 from the
n-terminus.  Contacts come from the document text, not from re-running
pyconfind, so the ground truth here is bit-identical to what the model was
trained to emit.
"""
from __future__ import annotations

import re

NUM_POS = 2000  # contacts-v1 position-token ring
MIN_SEP = 6     # contacts-v1 min_seq_separation
BEGIN = "<begin_statements>"

CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NTERM_RE = re.compile(r"<n-term>\s+<p(\d+)>")
RES_RE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")

_ONE_LETTER_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "X": "UNK",
}
THREE_TO_ONE = {three: one for one, three in _ONE_LETTER_TO_THREE.items()}


def parse_doc(doc: str):
    """``(L, one-letter sequence, sorted GT pairs in seq-index space)`` or ``None``.

    Returns ``None`` for anything that does not round-trip cleanly — a missing
    ``<n-term>``, or a ring index that carries no residue.  Callers treat that
    as "skip this row" rather than as an error; #222's corpora contain 2-residue
    peptides and other degenerate cases that are legal documents but useless
    training targets.

    Only the FIRST statements section is read (``rindex`` is deliberate in the
    generator's own header split, but here the prefix is everything before the
    first ``<begin_statements>``), so this is correct for plain single-section
    documents.  Multi-draft documents are built by us, never parsed by us.
    """
    if BEGIN not in doc:
        return None
    cut = doc.index(BEGIN) + len(BEGIN)
    prefix, struct = doc[:cut], doc[cut:]
    m = NTERM_RE.search(prefix)
    if not m:
        return None
    nterm = int(m.group(1))
    pos_in_seq = sorted({int(p) for p in re.findall(r"<p(\d+)>", prefix)},
                        key=lambda p: (p - nterm) % NUM_POS)
    seqidx = {p: (p - nterm) % NUM_POS for p in pos_in_seq}
    res_of_pos = {int(p): aa for p, aa in RES_RE.findall(prefix)}
    if not all(p in res_of_pos for p in pos_in_seq):
        return None
    seq = "".join(THREE_TO_ONE.get(res_of_pos[p], "X") for p in pos_in_seq)
    gt = set()
    for a, b in CONTACT_RE.findall(struct):
        ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        gt.add((min(ia, ib), max(ia, ib)))
    return len(pos_in_seq), seq, sorted(gt)
