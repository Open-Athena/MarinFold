# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""parse_contacts_ordered must be an order-preserving, logprob-alignable
companion to exp98's set-based parse_pred."""
import re

from rollout_metrics import parse_contacts_ordered, parse_pred

# pos_to_seq: position token index -> sequence index. Contrived so we can check
# the seq-sep >= 6 filter and the position->seq mapping.
POS_TO_SEQ = {100 + i: i for i in range(60)}  # <p100> -> 0, <p101> -> 1, ...


def _tok(text: str) -> list[str]:
    """Split a contacts-v1 statement string into single-token strings, matching
    convert_ids_to_tokens (every <...> is one token)."""
    return re.findall(r"<[^>]+>", text)


def test_ordered_matches_parse_pred_as_set():
    # emission order deliberately NOT sorted, with a duplicate and one too-short
    # (sep < 6) pair that must be dropped by both parsers.
    text = ("<begin_statements> "
            "<contact> <p130> <p110> "   # seq (30,10) sep 20  -> keep, normalized (10,30)
            "<contact> <p102> <p140> "   # seq (2,40)  sep 38  -> keep
            "<contact> <p130> <p110> "   # duplicate of first -> drop on dedup
            "<contact> <p105> <p108> "   # seq (5,8)   sep 3   -> drop (sep < 6)
            "<contact> <p120> <p150> "   # seq (20,50) sep 30  -> keep
            "<end>")
    toks = _tok(text)
    ordered = parse_contacts_ordered(toks, POS_TO_SEQ)
    pred_set = parse_pred(text, POS_TO_SEQ)

    # set equality with exp98's scorer path
    assert {(i, j) for i, j, _ in ordered} == pred_set
    # order preserved, first-occurrence dedup, short pair dropped
    assert [(i, j) for i, j, _ in ordered] == [(10, 30), (2, 40), (20, 50)]
    # k points at each statement's <contact> token
    for i, j, k in ordered:
        assert toks[k] == "<contact>"
        assert re.match(r"<p\d+>", toks[k + 1]) and re.match(r"<p\d+>", toks[k + 2])


def test_malformed_statement_skipped():
    # <contact> not followed by two position tokens must be skipped, not crash.
    text = "<contact> <p130> <ARG> <contact> <p102> <p140>"
    toks = _tok(text)
    ordered = parse_contacts_ordered(toks, POS_TO_SEQ)
    assert [(i, j) for i, j, _ in ordered] == [(2, 40)]


def test_truncated_tail_no_crash():
    # a <contact> at the very end with no operands (truncated generation).
    toks = _tok("<contact> <p102> <p140> <contact>")
    ordered = parse_contacts_ordered(toks, POS_TO_SEQ)
    assert [(i, j) for i, j, _ in ordered] == [(2, 40)]
