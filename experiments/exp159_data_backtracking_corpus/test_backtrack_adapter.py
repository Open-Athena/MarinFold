# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the exp120 adapter wiring (#159).

A fake Backend (real contacts-v1 tokenizer, no weights) plants the GT pairs so
the Scorer reads them high and false positives below the floor, and scripts the
Proposer's contact emissions. This validates the seq<->position mapping, prompt
assembly, next-contact parsing, and — the load-bearing check — that a full
engine run + ``assemble_document`` yields a real contacts-v1 document that
folds (via ``read.live_contacts``) to exactly GT. Retraction *dynamics* are
covered separately by ``test_backtrack_engine.py``.

Run from marinfold/::

    uv run pytest ../experiments/exp159_data_backtracking_corpus/test_backtrack_adapter.py -q
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from backtrack_adapter import ModelAdapter  # noqa: E402
from backtrack_engine import RetractionPolicy, build_backtracking_structure, canon  # noqa: E402

from marinfold import build_tokenizer  # noqa: E402
from marinfold.document_structures.contacts_v1 import inference as inf  # noqa: E402
from marinfold.document_structures.contacts_v1.read import (  # noqa: E402
    iter_structure_statements,
)
from marinfold.document_structures.contacts_v1.vocab import (  # noqa: E402
    all_domain_tokens,
    position_token,
)

_SEQ = "MGDIQVQVNIDDNGKAAAAQCDEFG"  # 25 residues
_ENTRY = "demo"


def _seq_positions():
    s = inf.structure_from_sequence(_SEQ, entry_id=_ENTRY)
    _, seq_positions, _ = inf._prefix_and_positions(s, entry_id=_ENTRY)
    return s, seq_positions


class FakeBackend:
    """Plants GT pairs high (Scorer) and scripts contact emissions (Proposer)."""

    def __init__(self, tokenizer, seq_positions, *, planted, script, low_val=1e-4):
        self._tok = tokenizer
        self._seq_positions = seq_positions
        self._pos_id_to_seq = {
            tokenizer.convert_tokens_to_ids(position_token(p)): k
            for k, p in enumerate(seq_positions)
        }
        self._planted = {frozenset(canon(*p)) for p in planted}
        self._in_planted = {i for pair in self._planted for i in pair}
        self._script = [canon(*p) for p in script]
        self._i = 0
        self._low = low_val

    @property
    def tokenizer(self):
        return self._tok

    def next_token_probs(self, prefix_token_ids, tail_token_ids_batch, target_token_ids):
        col_seq = [self._pos_id_to_seq[t] for t in target_token_ids]
        out = np.full((len(tail_token_ids_batch), len(target_token_ids)), self._low)
        for row, tail in enumerate(tail_token_ids_batch):
            if len(tail) == 1:  # lp1
                for col, m in enumerate(col_seq):
                    if m in self._in_planted:
                        out[row, col] = 0.5
            else:               # lp2 conditioned on tail[1]
                cond = self._pos_id_to_seq[tail[1]]
                for col, m in enumerate(col_seq):
                    if frozenset({cond, m}) in self._planted:
                        out[row, col] = 0.9
        return out

    def sample_completions(self, prefix_token_ids_batch, **kwargs):
        completions = []
        for prefix_ids in prefix_token_ids_batch:
            text = self._tok.decode(prefix_ids, skip_special_tokens=False)
            live = set()
            for kind, a, b in iter_structure_statements(text):
                ia = self._pos_id_to_seq.get(
                    self._tok.convert_tokens_to_ids(position_token(a)))
                ib = self._pos_id_to_seq.get(
                    self._tok.convert_tokens_to_ids(position_token(b)))
                if ia is not None and ib is not None:
                    live.add(canon(ia, ib))
            while self._i < len(self._script) and self._script[self._i] in live:
                self._i += 1
            if self._i < len(self._script):
                i, j = self._script[self._i]
                self._i += 1
                toks = ["<contact>", position_token(self._seq_positions[i]),
                        position_token(self._seq_positions[j])]
                completions.append(
                    list(self._tok.encode(" ".join(toks), add_special_tokens=False)))
            else:
                completions.append(
                    list(self._tok.encode("<end>", add_special_tokens=False)))
        return completions


def _make(planted, script):
    structure, seq_positions = _seq_positions()
    backend = FakeBackend(
        build_tokenizer(all_domain_tokens()), seq_positions,
        planted=planted, script=script,
    )
    adapter = ModelAdapter(backend, structure, entry_id=_ENTRY)
    return adapter


def test_propose_round_trips_positions_to_seq():
    gt = {(0, 12), (2, 16)}
    adapter = _make(gt, script=[(0, 12), (2, 16)])
    assert adapter.propose([]) == canon(0, 12)
    assert adapter.propose([canon(0, 12)]) == canon(2, 16)


def test_score_gt_above_false_positive():
    gt = frozenset({canon(0, 12), canon(2, 16), canon(5, 20)})
    adapter = _make(gt, script=[])
    s = adapter.score(
        committed=[canon(0, 12), canon(2, 16)],
        targets=[canon(5, 20), canon(1, 13)],   # a GT pair vs a false positive
    )
    assert s[canon(5, 20)] > s[canon(1, 13)]
    assert s[canon(1, 13)] < 1e-3               # below the default floor


def test_full_run_assembles_document_folding_to_gt():
    gt = frozenset({canon(0, 12), canon(2, 16), canon(5, 20)})
    # Model proposes the GT pairs interleaved with two false positives.
    script = [(0, 12), (1, 13), (2, 16), (3, 18), (5, 20)]
    adapter = _make(gt, script=script)
    policy = RetractionPolicy(min_delay=1, eval_cadence=1, s_floor=1e-3)

    res = build_backtracking_structure(
        gt, adapter, adapter, policy, max_statements=200, rng=random.Random(0)
    )
    doc = adapter.assemble_document(res.statements)

    assert res.correct
    assert adapter.document_folds_to_gt(doc, gt)   # the rendered doc == GT
    assert doc.startswith("<contacts-v1>") and doc.strip().endswith("<end>")
    # The two false positives were emitted then retracted (present as <retract>).
    assert res.n_retract_statements >= 2
    # Sanity: the document actually contains retract statements.
    assert "<retract>" in doc
