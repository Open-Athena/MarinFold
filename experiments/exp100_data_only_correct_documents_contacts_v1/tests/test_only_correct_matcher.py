# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The incremental OnlyCorrectMatcher must agree with the replay-based
legal_token_ids at every step, and the bitmask packing must round-trip with the
TPU unpack convention (bit==1 -> allowed)."""
from __future__ import annotations

import random

import numpy as np

from constrained_grammar import (
    OnlyCorrectMatcher,
    build_remaining,
    legal_token_ids,
    pack_allowed_bitmask,
)

CONTACT = 5
END = 10
GT = [(100, 200), (200, 300), (150, 400), (300, 500)]


def test_matcher_matches_legal_token_ids_random_walk():
    rng = random.Random(1)
    for _ in range(300):
        m = OnlyCorrectMatcher(GT, contact_id=CONTACT, end_id=END)
        gen: list[int] = []
        for _ in range(200):
            replay = legal_token_ids(gen, contact_id=CONTACT, end_id=END,
                                     remaining_full=build_remaining(GT))
            assert m.allowed() == replay, (gen, m.allowed(), replay)
            tok = rng.choice(sorted(m.allowed()))
            assert m.accept(tok)
            gen.append(tok)
            if tok == END:
                break
        assert m.terminated and gen[-1] == END
        assert m.allowed() == set()  # nothing allowed after termination
        assert (len(gen) - 1) // 3 == len(build_remaining(GT))


def test_accept_rejects_illegal():
    m = OnlyCorrectMatcher(GT, contact_id=CONTACT, end_id=END)
    assert not m.accept(END)      # can't end before emitting contacts
    assert not m.accept(999)      # not <contact>
    assert m.accept(CONTACT)
    assert not m.accept(CONTACT)  # need a position now


def _unpack(row_int32, vocab):
    """Mirror the TPU structured_decode_fn unpack: bit==0 -> disallowed."""
    words = row_int32.view(np.uint32)
    arange = np.arange(32, dtype=np.uint32)
    bits = (words[:, None] >> arange[None, :]) & 1  # [n_words, 32]
    flat = bits.reshape(-1)[:vocab]
    return set(np.nonzero(flat == 1)[0].tolist())


def test_bitmask_pack_roundtrip():
    vocab = 2846
    n_words = (vocab + 31) // 32
    for allowed in ([CONTACT], [END], [100, 200, 300, 500, 150, 400], [31, 32, 63, 2845]):
        row = pack_allowed_bitmask(allowed, n_words)
        assert _unpack(row, vocab) == set(allowed)
