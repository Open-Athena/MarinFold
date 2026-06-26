# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the two correctness-critical KNN helpers (no network/mmseqs).

* ``parse_document`` — recover sequence + contacts from a synthetic document with
  a known wrap-around N-terminus and resampled token order.
* ``target_to_query_map`` — walk a hand-built ``qaln``/``taln`` pair with gaps on
  both sides into the expected 0-based residue map.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knn_lib import MIN_SEP, parse_document, target_to_query_map  # noqa: E402


def test_parse_document_wraparound_and_resampled_order():
    # seq_len=5, n_term_index=1998 -> positions 1998,1999,0,1,2 map to seqpos 0..4.
    # Residues emitted out of order; one contact wraps across the modulo boundary.
    nti = 1998
    seq = "MKLVG"  # MET LYS LEU VAL GLY at seqpos 0..4
    three = {"M": "MET", "K": "LYS", "L": "LEU", "V": "VAL", "G": "GLY"}
    toks = [(1998 + k) % 2000 for k in range(5)]  # [1998,1999,0,1,2]
    order = [3, 0, 4, 1, 2]  # resampled emission order
    seq_tokens = " ".join(f"<p{toks[k]}> <{three[seq[k]]}>" for k in order)
    # contact between seqpos 0 (tok 1998) and seqpos 4 (tok 2): separation 4.
    doc = (f"<contacts-v1> <begin_sequence> {seq_tokens} "
           f"<begin_statements> <contact> <p{toks[0]}> <p{toks[4]}>")

    parsed_seq, contacts = parse_document(doc, seq_len=5, n_term_index=nti)
    assert parsed_seq == seq
    assert contacts == [(0, 4)]  # 0-based, sorted, both in range


def test_parse_document_drops_out_of_range_and_unknown():
    # seq_len=3, anchor 0. A stray token at seqpos 7 (out of range) is ignored;
    # an unknown residue becomes X.
    doc = ("<begin_sequence> <p0> <ALA> <p1> <XYZ> <p2> <GLY> <p7> <LEU> "
           "<begin_statements> <contact> <p0> <p7>")  # contact references oor pos
    seq, contacts = parse_document(doc, seq_len=3, n_term_index=0)
    assert seq == "AXG"
    assert contacts == []  # the only contact references an out-of-range residue


def test_target_to_query_map_with_gaps():
    # query residues 5.. , target residues 10.. (1-based starts -> 4 / 9 zero-based).
    #   qaln: A B - C D
    #   taln: A B C - D
    # col0: q4<->t9 ; col1: q5<->t10 ; col2: gap in query (t11 consumed, no pair)
    # col3: gap in target (q6 consumed, no pair) ; col4: q7<->t12
    t2q = target_to_query_map("AB-CD", "ABC-D", qstart=5, tstart=10)
    assert t2q == {9: 4, 10: 5, 12: 7}
    assert 11 not in t2q  # target residue aligned to a query gap


def test_min_sep_constant():
    # Guard the separation threshold the voting loop relies on.
    assert MIN_SEP == 6
