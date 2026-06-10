# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contacts-v1 tests that actually run pyconfind.

These need ``pyconfind`` installed (``uv sync --extra contacts-v1``) and,
on a cold cache, network access to download the Dunbrack rotamer library
once — hence the ``network`` marker. Skip with ``pytest -m 'not network'``.
"""

from pathlib import Path

import pytest

pytest.importorskip("pyconfind")

from marinfold import build_tokenizer  # noqa: E402
from marinfold.document_structures.contacts_v1 import (  # noqa: E402
    CONTEXT_LENGTH,
    all_domain_tokens,
    analyze_structure,
    generate_document,
)

_1QYS = Path(__file__).parents[2] / "data" / "1QYS.cif"


@pytest.mark.network
def test_analyze_1qys():
    analyzed = analyze_structure(_1QYS)
    assert analyzed.entry_id == "1QYS"
    assert len(analyzed.residues) == 92
    assert {r.chain for r in analyzed.residues} == {"A"}
    # MSE in 1QYS is canonicalized to MET.
    assert all(r.resname != "MSE" for r in analyzed.residues)
    assert "MET" in {r.resname for r in analyzed.residues}
    # Contacts: degree > 0, lower-triangular, sorted by (seq_i, seq_j).
    assert len(analyzed.contacts) > 0
    for c in analyzed.contacts:
        assert c.seq_i < c.seq_j
        assert c.degree > 0
    pairs = [(c.seq_i, c.seq_j) for c in analyzed.contacts]
    assert pairs == sorted(pairs)


@pytest.mark.network
def test_generate_document_1qys_tokenizes_cleanly():
    res = generate_document(_1QYS)
    assert res is not None
    assert res.entry_id == "1QYS"
    assert res.seq_len == 92
    assert res.contacts_pre_filter > 0
    # 1QYS has a long tail of near-zero contacts; the 0.001 filter drops them.
    assert res.contacts_passing_min_degree < res.contacts_pre_filter
    assert res.contacts_emitted == res.contacts_passing_min_degree  # all eligible fit
    assert res.contacts_excluded == res.contacts_pre_filter - res.contacts_emitted
    assert res.truncated is False  # eligible all fit; weak ones filtered, not truncated
    assert res.num_tokens <= CONTEXT_LENGTH
    # Every emitted contact is above threshold; whole-protein min is below it.
    degrees = [c.degree for c in res.contacts]
    assert all(d >= 0.001 for d in degrees)
    assert res.highest_contact_degree == max(degrees)     # strongest is included
    assert res.lowest_included_contact_degree == min(degrees) >= 0.001
    assert res.lowest_nonzero_contact_degree < 0.001      # raw min is below threshold
    # Tokenizes 1:1 with the published vocab, no UNK collapse.
    tok = build_tokenizer(all_domain_tokens())
    ids = tok.encode(res.document, add_special_tokens=False)
    assert len(ids) == len(res.document.split())
    assert tok.convert_tokens_to_ids("<UNK>") not in ids


@pytest.mark.network
def test_generate_document_is_deterministic():
    a = generate_document(_1QYS)
    b = generate_document(_1QYS)
    assert a is not None and b is not None
    assert a.document == b.document
