# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the only-correct contacts-v1 decoding grammar (no model)."""
from __future__ import annotations

import random

import pytest

from constrained_grammar import (
    ContactConstraint,
    build_remaining,
    legal_token_ids,
)

CONTACT = 3   # stand-in token ids; values are arbitrary for the FSM
END = 1
# GT contacts as position-token ids. Pick ids disjoint from CONTACT/END.
GT = [(100, 200), (200, 300), (150, 400)]


def _legal(gen):
    return legal_token_ids(gen, contact_id=CONTACT, end_id=END,
                           remaining_full=build_remaining(GT))


def test_start_forces_contact_when_remaining():
    assert _legal([]) == {CONTACT}


def test_first_endpoint_is_any_remaining_endpoint():
    # after <contact>, any position that participates in a remaining contact
    assert _legal([CONTACT]) == {100, 200, 300, 150, 400}


def test_second_endpoint_completes_pending_pair():
    # chose pi=200 -> only 100 or 300 complete a true contact (200 touches both)
    assert _legal([CONTACT, 200]) == {100, 300}
    # chose pi=150 -> only 400
    assert _legal([CONTACT, 150]) == {400}
    # chose pi=100 -> only 200
    assert _legal([CONTACT, 100]) == {200}


def test_orientation_is_symmetric():
    # emitting the pair in the reverse order is allowed and consumes the contact
    gen = [CONTACT, 200, 100]  # = contact {100,200} written 200-first
    # next statement start: 2 contacts remain -> <contact>
    assert _legal(gen) == {CONTACT}
    assert _legal(gen + [CONTACT]) == {200, 300, 150, 400}  # 100 gone


def test_contact_consumed_only_once():
    gen = [CONTACT, 100, 200]  # consumed {100,200}
    assert _legal(gen + [CONTACT, 200]) == {300}  # 200-100 no longer available


def test_end_masked_until_all_emitted_then_forced():
    g = [CONTACT, 100, 200, CONTACT, 200, 300]  # 2 of 3 done
    assert _legal(g) == {CONTACT}  # still one left, no <end>
    g2 = g + [CONTACT, 150, 400]   # all 3 done
    assert _legal(g2) == {END}     # <end> forced, nothing else


def test_replay_rejects_invalid_stream():
    with pytest.raises(ValueError):
        _legal([CONTACT, 100, 999])  # 100-999 is not a true contact


def test_max_new_tokens():
    c = ContactConstraint([7, 8, 9], GT, contact_id=CONTACT, end_id=END)
    assert c.max_new_tokens() == 3 * 3 + 1  # 3 contacts -> 10 tokens incl <end>


def test_offset_autodetect_generated_only():
    # past_token_ids contains ONLY generated tokens (offset 0)
    c = ContactConstraint([7, 8, 9], GT, contact_id=CONTACT, end_id=END)
    assert c._generated([CONTACT, 100]) == [CONTACT, 100]
    assert c._offset == 0


def test_offset_autodetect_prompt_prepended():
    # past_token_ids starts with the prompt -> it is sliced off
    prompt = [7, 8, 9]
    c = ContactConstraint(prompt, GT, contact_id=CONTACT, end_id=END)
    assert c._generated(prompt + [CONTACT, 100]) == [CONTACT, 100]
    assert c._offset == 3


def test_full_random_walk_produces_valid_permutation():
    """Drive the FSM like a sampler: always pick a legal token; must end clean."""
    rng = random.Random(0)
    for _ in range(200):
        gen: list[int] = []
        for _step in range(100):
            allowed = _legal(gen)
            tok = rng.choice(sorted(allowed))
            gen.append(tok)
            if tok == END:
                break
        assert gen[-1] == END
        # exactly the 3 true contacts, each once, in some order/orientation
        emitted = {frozenset((gen[3 * k + 1], gen[3 * k + 2]))
                   for k in range((len(gen) - 1) // 3)}
        assert emitted == build_remaining(GT)
        assert len(gen) == 3 * len(GT) + 1
