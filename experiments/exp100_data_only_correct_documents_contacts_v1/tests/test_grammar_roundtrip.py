# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline integration test: seq-index GT -> position tokens -> FSM-driven
constrained stream -> rendered document text -> parse_pred, and confirm the
recovered contacts exactly equal the GT (recall == precision == 1.0). Exercises
the worker's coordinate mapping + the grammar + the scorer's parser together,
with no model. The key real-world failure mode is a position<->seq mapping slip;
this catches it."""
from __future__ import annotations

import random

from constrained_grammar import build_remaining, legal_token_ids
from rollout_metrics import parse_pred

NUM_POS = 2000
CONTACT = -1  # sentinels distinct from any position id
END = -2


def _walk(gt_pos_ids, rng):
    """Random-walk the FSM to a complete only-correct stream of position ids."""
    remaining = build_remaining(gt_pos_ids)
    gen: list[int] = []
    while True:
        allowed = legal_token_ids(gen, contact_id=CONTACT, end_id=END,
                                  remaining_full=remaining)
        gen.append(rng.choice(sorted(allowed)))
        if gen[-1] == END:
            return gen


def _render(gen):
    """Render an FSM stream (position ids = ints) as contacts-v1 structure text."""
    parts = []
    for t in gen:
        if t == CONTACT:
            parts.append("<contact>")
        elif t == END:
            parts.append("<end>")
        else:
            parts.append(f"<p{t}>")
    return " ".join(parts)


def test_roundtrip_random_realizations():
    rng = random.Random(7)
    for _ in range(300):
        L = rng.randint(20, 300)
        n_term = rng.randrange(NUM_POS)
        # seq index i -> wrap-around position (matches gen_prompts.seq_positions)
        seq_positions = [(n_term + i) % NUM_POS for i in range(L)]
        pos_to_seq = {pos: i for i, pos in enumerate(seq_positions)}
        # random GT contacts in seq space, seq-sep >= 6 (the contacts-v1 def)
        gt_seq = set()
        for _ in range(rng.randint(5, 40)):
            i = rng.randrange(L)
            j = rng.randrange(L)
            if abs(i - j) >= 6:
                gt_seq.add((min(i, j), max(i, j)))
        if not gt_seq:
            continue
        gt_pos_ids = [(seq_positions[i], seq_positions[j]) for i, j in gt_seq]
        gen = _walk(gt_pos_ids, rng)
        text = _render(gen)
        pred = parse_pred(text, pos_to_seq)
        assert pred == gt_seq, f"roundtrip mismatch (n_term={n_term}, L={L})"
        # exactly |gt| contacts, full recall by construction
        assert (len(gen) - 1) // 3 == len(gt_seq)
