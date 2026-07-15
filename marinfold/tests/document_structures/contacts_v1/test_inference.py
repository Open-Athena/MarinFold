# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for contacts-v1 inference (predict / evaluate) — no model download.

The model forward pass is replaced by ``_PlantedBackend``, a stub
:class:`~marinfold.Backend` whose ``next_token_probs`` boosts a chosen set of
sequence-index contacts. That lets us check the full pairwise-readout
pipeline (prefix construction → ``P(contact)`` → ranking → metrics) on the
CPU, with a real contacts-v1 tokenizer but no weights.

Run from the marinfold/ dir::

    uv sync
    uv run pytest tests/document_structures/contacts_v1/test_inference.py -v
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import inference as inf
from marinfold.document_structures.contacts_v1.parse import RawContact
from marinfold.document_structures.contacts_v1.vocab import (
    all_domain_tokens,
    position_token,
)


_SEQ = "MGDIQVQVNIDDNGKAAAAQ"  # 20 residues — long enough for sep-12 contacts


def _tokenizer():
    return build_tokenizer(all_domain_tokens())


class _PlantedBackend:
    """Stub backend: planted (seq_i, seq_j) contacts get boosted probability.

    ``next_token_probs`` returns softmax-style mass over the target position
    tokens. For a 1-token tail (``[<contact>]``, i.e. ``lp1``) every position
    that participates in a planted contact is boosted; for a 2-token tail
    (``[<contact>, <p_i>]``, i.e. ``lp2[i, :]``) the partner position of a
    planted contact with ``i`` is boosted. So planted pairs dominate the
    resulting ``P(contact)`` map.
    """

    def __init__(self, tokenizer, seq_positions, planted=()):
        self._tok = tokenizer
        self._pos_id_to_seq = {
            tokenizer.convert_tokens_to_ids(position_token(p)): k
            for k, p in enumerate(seq_positions)
        }
        self._planted = {frozenset(pair) for pair in planted}
        self._in_planted = {i for pair in self._planted for i in pair}

    @property
    def tokenizer(self):
        return self._tok

    def next_token_probs(self, prefix_token_ids, tail_token_ids_batch, target_token_ids):
        col_seq = [self._pos_id_to_seq[t] for t in target_token_ids]
        out = np.full((len(tail_token_ids_batch), len(target_token_ids)), 1e-3)
        for row, tail in enumerate(tail_token_ids_batch):
            if len(tail) == 1:  # lp1
                for col, m in enumerate(col_seq):
                    if m in self._in_planted:
                        out[row, col] = 0.5
            else:  # lp2, conditioned on the position in tail[1]
                cond = self._pos_id_to_seq[tail[1]]
                for col, m in enumerate(col_seq):
                    if frozenset({cond, m}) in self._planted:
                        out[row, col] = 0.9
        return out

    def sample_completions(self, prefix_token_ids_batch, *, max_new_tokens,
                           temperature=1.0, top_p=0.95, top_k=50,
                           stop_token_id=None, seed=None, batch_size=None):
        """Emit each prefix's planted contacts, in *that* realization's tokens.

        Each rollout prefix is a freshly resampled realization (different
        N-terminus + statement order). We recover its position numbering from
        the decoded prefix and write ``<contact> <p_a> <p_b>`` for every planted
        seq-index pair, so the rollout vote-counter recovers them — exercising
        the real resample → parse → vote path.
        """
        completions = []
        for prefix_ids in prefix_token_ids_batch:
            text = self._tok.decode(prefix_ids, skip_special_tokens=False)
            nterm = int(re.search(r"<n-term>\s+<p(\d+)>", text).group(1))
            positions = sorted(
                {int(p) for p in re.findall(r"<p(\d+)>", text)},
                key=lambda p: (p - nterm) % 2000,
            )
            toks: list[str] = []
            for pair in self._planted:
                a, b = sorted(pair)
                toks += ["<contact>", f"<p{positions[a]}>", f"<p{positions[b]}>"]
            completions.append(
                list(self._tok.encode(" ".join(toks), add_special_tokens=False))
            )
        return completions


# ---------------------------------------------------------------------------
# structure_from_sequence + prefix construction
# ---------------------------------------------------------------------------


def test_structure_from_sequence_basic():
    s = inf.structure_from_sequence("ACDEFGHIKL")
    assert len(s.residues) == 10
    assert s.gt_contacts is None
    assert s.residues[0].resname == "ALA"
    assert s.entry_id == "sequence"


def test_structure_from_sequence_too_short_raises():
    with pytest.raises(ValueError):
        inf.structure_from_sequence("A")


def test_prefix_ends_with_begin_statements_and_is_deterministic():
    s = inf.structure_from_sequence(_SEQ, entry_id="x")
    prefix, positions, seq_len = inf._prefix_and_positions(s, entry_id="x")
    again = inf._prefix_and_positions(s, entry_id="x")
    assert prefix.endswith("<begin_statements>")
    assert prefix.startswith("<contacts-v1> <begin_sequence>")
    assert seq_len == len(s.residues)
    assert positions == again[1]  # deterministic given the entry id
    # A different ensemble salt reshuffles the numbering.
    assert inf._prefix_and_positions(s, entry_id="x#cv1ens1")[1] != positions


def test_token_id_rejects_unk_collapse():
    class _Tok:
        unk_token_id = 7

        def convert_tokens_to_ids(self, _token):
            return 7

    with pytest.raises(ValueError, match="contacts-v1"):
        inf._token_id(_Tok(), "<contact>")


# ---------------------------------------------------------------------------
# P(contact) scoring
# ---------------------------------------------------------------------------


def test_pcontact_matrix_symmetric_and_planted_pair_wins():
    s = inf.structure_from_sequence(_SEQ, entry_id="demo")
    prefix, seq_positions, seq_len = inf._prefix_and_positions(s, entry_id="demo")
    backend = _PlantedBackend(_tokenizer(), seq_positions, planted=[(0, 12)])
    pcontact = inf._pcontact_matrix(backend, prefix, seq_positions)

    assert pcontact.shape == (seq_len, seq_len)
    assert np.allclose(pcontact, pcontact.T)  # unordered ⇒ symmetric
    candidates = inf._candidate_pairs(seq_len, 6)
    best = max(candidates, key=lambda ij: pcontact[ij])
    assert best == (0, 12)


# ---------------------------------------------------------------------------
# Metrics (pure numpy, no backend)
# ---------------------------------------------------------------------------


def test_rank_auc_known_cases():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    labels = np.array([1, 1, 0, 0], dtype=bool)
    assert inf._rank_auc(scores, labels) == 1.0
    assert inf._rank_auc(-scores, labels) == 0.0
    # All tied ⇒ chance.
    assert inf._rank_auc(np.ones(4), np.array([1, 1, 0, 0], dtype=bool)) == 0.5
    # One class absent ⇒ undefined.
    assert math.isnan(inf._rank_auc(np.array([1.0, 2.0]), np.array([1, 1], dtype=bool)))


def test_gt_contact_matrix_filters_degree_and_separation():
    contacts = [
        RawContact(0, 12, 0.5),     # kept (sep 12, degree ok)
        RawContact(0, 3, 0.9),      # dropped: sep 3 < 6
        RawContact(1, 14, 0.0005),  # dropped: degree < 0.001
    ]
    gt = inf._gt_contact_matrix(contacts, 16, 6)
    assert gt[0, 12] and gt[12, 0]
    assert not gt[0, 3]
    assert not gt[1, 14]


def test_metric_rows_perfect_ranking():
    seq_len = 14
    truth = [(0, 6), (1, 8), (2, 13)]  # all short-range (sep 6/7/11)
    gt = np.zeros((seq_len, seq_len), dtype=bool)
    pcontact = np.full((seq_len, seq_len), 0.01)
    for i, j in truth:
        gt[i, j] = gt[j, i] = True
        pcontact[i, j] = pcontact[j, i] = 1.0  # true contacts score highest
    rows = inf._metric_rows(pcontact, gt, seq_len, 6)
    assert rows["all"]["auc"] == 1.0
    assert rows["all"]["r_precision"] == 1.0
    assert rows["short"]["r_precision"] == 1.0


# ---------------------------------------------------------------------------
# predict / evaluate end to end (stub backend)
# ---------------------------------------------------------------------------


def test_predict_record_shape_and_ranking(monkeypatch):
    s = inf.structure_from_sequence(_SEQ, entry_id="demo")
    prefix, seq_positions, seq_len = inf._prefix_and_positions(s, entry_id="demo")
    backend = _PlantedBackend(_tokenizer(), seq_positions, planted=[(0, 12)])
    monkeypatch.setattr(inf, "_make_backend", lambda cfg: backend)

    cfg = inf.InferenceConfig(model="/stub", backend="transformers", keep_matrix=True)
    records = list(inf.predict(cfg, structures=[s]))
    assert len(records) == 1
    rec = records[0]
    assert rec["entry_id"] == "demo"
    assert rec["n_residues"] == seq_len
    assert rec["min_seq_separation"] == 6
    assert rec["method"] == "pairwise"
    n_candidates = len(inf._candidate_pairs(seq_len, 6))
    assert len(rec["pairs"]) == n_candidates == len(rec["score"])
    for i, j in rec["pairs"]:
        assert 1 <= i < j <= seq_len and (j - i) >= 6  # 1-indexed candidates
    top = rec["pairs"][int(np.argmax(rec["score"]))]
    assert tuple(top) == (1, 13)  # planted (0,12) → 1-indexed (1,13)

    matrix = np.array(rec["score_matrix"])
    assert matrix.shape == (seq_len, seq_len)
    assert np.isnan(matrix[0, 0])  # diagonal / band blanked


def test_predict_empty_short_circuits():
    cfg = inf.InferenceConfig(model="/nonexistent", backend="transformers")
    assert list(inf.predict(cfg, structures=[])) == []


def test_evaluate_recovers_planted_contacts(monkeypatch):
    s0 = inf.structure_from_sequence(_SEQ, entry_id="demo")
    prefix, seq_positions, _ = inf._prefix_and_positions(s0, entry_id="demo")
    planted = [(0, 12), (2, 16)]
    backend = _PlantedBackend(_tokenizer(), seq_positions, planted=planted)
    monkeypatch.setattr(inf, "_make_backend", lambda cfg: backend)

    structure = inf.ContactStructure(
        entry_id="demo",
        residues=s0.residues,
        gt_contacts=tuple(RawContact(i, j, 1.0) for i, j in planted),
    )
    cfg = inf.InferenceConfig(model="/stub", backend="transformers")
    result = inf.evaluate(cfg, structures=[structure])

    assert result.extras["n_structures"] == 1
    assert result.extras["method"] == "pairwise"
    assert result.metrics["auc_all"] == 1.0
    assert result.metrics["r_precision_all"] == 1.0
    assert any(row["gt"] == 1 for row in result.per_example)
    assert all(
        set(row) >= {"entry_id", "i", "j", "score", "gt"}
        for row in result.per_example
    )


def test_evaluate_empty_returns_warning():
    cfg = inf.InferenceConfig(model="/nonexistent", backend="transformers")
    result = inf.evaluate(cfg, structures=[])
    assert result.metrics == {}
    assert result.per_example == []
    assert result.extras.get("warning") == "no input structures"


# ---------------------------------------------------------------------------
# rollout method
# ---------------------------------------------------------------------------


def test_tiebreak_keeps_votes_primary():
    votes = np.array([[0, 2, 0], [2, 0, 5], [0, 5, 0]], dtype=float)
    sym = np.array([[0, 9, 1], [9, 0, -3], [1, -3, 0]], dtype=float)
    combined = inf._tiebreak(votes, sym)
    # The 5-vote pair outranks the 2-vote pair regardless of pairwise.
    assert combined[1, 2] > combined[0, 1]
    # The pairwise offset stays in [0, 0.5): it never crosses a vote boundary.
    offset = combined - votes
    assert offset.min() >= 0.0
    assert offset.max() <= 0.5 + 1e-9


def test_score_matrix_rejects_unknown_method():
    s = inf.structure_from_sequence(_SEQ, entry_id="demo")
    cfg = inf.InferenceConfig(model="/x", method="bogus")
    with pytest.raises(ValueError, match="pairwise|rollout"):
        inf._score_matrix(object(), s, cfg)


def test_predict_rollout_votes_and_ranks(monkeypatch):
    s = inf.structure_from_sequence(_SEQ, entry_id="demo")
    canonical_positions = inf._prefix_and_positions(s, entry_id="demo")[1]
    backend = _PlantedBackend(_tokenizer(), canonical_positions, planted=[(0, 12)])
    monkeypatch.setattr(inf, "_make_backend", lambda cfg: backend)

    cfg = inf.InferenceConfig(
        model="/stub", backend="transformers", method="rollout", n_rollouts=4
    )
    records = list(inf.predict(cfg, structures=[s]))
    assert len(records) == 1
    rec = records[0]
    assert rec["method"] == "rollout"
    assert rec["n_rollouts"] == 4
    assert "ensemble_k" not in rec
    top = rec["pairs"][int(np.argmax(rec["score"]))]
    assert tuple(top) == (1, 13)  # planted (0,12) wins the vote
    # Every rollout votes the planted pair, so its score is >= n_rollouts.
    assert max(rec["score"]) >= 4.0


def test_evaluate_rollout_recovers_planted(monkeypatch):
    s0 = inf.structure_from_sequence(_SEQ, entry_id="demo")
    canonical_positions = inf._prefix_and_positions(s0, entry_id="demo")[1]
    planted = [(0, 12), (2, 16)]
    backend = _PlantedBackend(_tokenizer(), canonical_positions, planted=planted)
    monkeypatch.setattr(inf, "_make_backend", lambda cfg: backend)

    structure = inf.ContactStructure(
        entry_id="demo",
        residues=s0.residues,
        gt_contacts=tuple(RawContact(i, j, 1.0) for i, j in planted),
    )
    cfg = inf.InferenceConfig(
        model="/stub", backend="transformers", method="rollout", n_rollouts=3
    )
    result = inf.evaluate(cfg, structures=[structure])
    assert result.extras["method"] == "rollout"
    assert result.metrics["r_precision_all"] == 1.0
