# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Invariants the corpus has to satisfy, as tests rather than as prose.

Run against the staged PDB shards (the smallest arm):

    /data/exp208_replication/venv/bin/python -m pytest test_corpus.py -q
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

import corpus_sources as cs
from _contacts_v1_doc import parse_doc
from _loss_mask import span_weights
from build_corpus import MULTI_DOC_TOKEN, PLAIN_DOC_TOKEN, build_multi, build_plain

WORK = Path("/data/exp230_multi")
N = 100


def sample_rows(n=N):
    if not cs.manifest_path(WORK).exists():
        pytest.skip(f"no staged corpus at {WORK}")
    return list(itertools.islice(
        cs.iter_corpus_rows(cs.PDB_MONOMERS, work=WORK, log=lambda *a: None), n))


def test_plain_regeneration_is_lossless():
    """A regenerated rehearsal document carries the corpus document's contacts.

    The rehearsal half is rebuilt by the library generator from ground truth
    parsed out of the published document, rather than reusing the published
    text, so that both halves of the corpus come from one protein pool with
    freshly drawn realizations. That is only sound if the rebuild is lossless.
    """
    rows = sample_rows()
    assert rows, "no usable PDB rows"
    for rec in rows:
        gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
        built = build_plain(f"{rec['entry_id']}:p0", rec["sequence"], sorted(gt))
        assert built is not None
        doc, _ = built
        parsed = parse_doc(doc)
        assert parsed is not None
        L, seq, gt2 = parsed
        assert seq == rec["sequence"]
        assert L == rec["L"]
        assert set(gt2) == gt


def test_multi_document_shape():
    """K drafts + one final section, mode marker at position 0, closed by <end>."""
    rng = np.random.default_rng(0)
    for rec in sample_rows(40):
        gt = [(int(i), int(j)) for i, j in rec["gt_contacts"]]
        # Two synthetic "rollouts": a subset of truth and a shifted decoy, which
        # is enough to exercise the draft path without needing real generations.
        rollouts = [
            [v for pair in gt[::2] for v in pair],
            [v for pair in [(i, j) for i, j in gt if i + 1 < j] for v in pair],
        ]
        built = build_multi(f"{rec['entry_id']}:m0", rec["sequence"], gt, rollouts,
                            [0.3, 0.2], kmax=4, n_cap=50, rng=rng)
        if built is None:
            continue
        doc, meta = built
        toks = doc.split()
        assert toks[0] == MULTI_DOC_TOKEN
        assert toks[-1] == "<end>"
        assert doc.count("<begin_statements>") == meta["K"] + 1
        assert doc.count("<end>") == 1
        assert meta["n_tokens"] <= 8192


def test_plain_document_has_one_section():
    for rec in sample_rows(40):
        gt = [(int(i), int(j)) for i, j in rec["gt_contacts"]]
        built = build_plain(f"{rec['entry_id']}:p0", rec["sequence"], gt)
        assert built is not None
        doc, _ = built
        assert doc.split()[0] == PLAIN_DOC_TOKEN
        assert doc.count("<begin_statements>") == 1
        assert doc.count("<end>") == 1


def test_profile_f_weights_the_restart_and_stop_decisions_equally():
    """The mechanical fact #163 established, pinned.

    ``weight[i]`` supervises predicting ``token[i+1]``, so the slot that teaches
    "emit another ``<begin_statements>``" is the LAST token of a draft and the
    slot that teaches "emit ``<end>``" is the last token of the final section.
    Under profile F those must be equal -- that equality is what makes
    continuing competitive with stopping, and it is the difference between
    #163's collapsed arms and its published one.
    """
    # <contacts-v1.multi> <begin_sequence> ... <begin_statements> d1
    # <begin_statements> d2 <begin_statements> final <end>
    ids = np.array([7, 8, 100, 101, 9, 5, 143, 153, 9, 5, 143, 154, 9, 5, 143, 155, 10])
    w = span_weights(ids, w_header=0.1, w_draft=1.0, w_final=1.0)
    begins = np.flatnonzero(ids == 9)
    end = int(np.flatnonzero(ids == 10)[0])
    # last token of draft 1 -> predicts the second <begin_statements>
    restart = float(w[begins[1] - 1])
    # last token of the final section -> predicts <end>
    stop = float(w[end - 1])
    assert restart == stop == 1.0
    assert float(w[0]) == pytest.approx(0.1)   # header
    assert float(w[end]) == pytest.approx(0.1)  # the <end> slot itself
