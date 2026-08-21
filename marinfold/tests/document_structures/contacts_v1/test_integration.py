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
    GenerationConfig,
    all_domain_tokens,
    analyze_structure,
    chain_segments,
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


# --- Multi-chain (multimer) end-to-end -----------------------------------
#
# 3I40 is human insulin: two short chains (A, 21 residues; B, 30) that exist
# only as a heterodimer, so it is the smallest honest test of the multimer
# path -- an interface is the whole point of the structure, and the ASU needs
# no assembly expansion to show one.
_3I40 = Path(__file__).parents[2] / "data" / "3I40.cif"


@pytest.mark.network
def test_analyze_multi_chain_requires_opt_in():
    """A multi-chain input is rejected under the default single-chain policy."""
    with pytest.raises(ValueError, match="max_chains=1"):
        analyze_structure(_3I40)

    analyzed = analyze_structure(_3I40, max_chains=2)
    assert [(s.chain, s.length) for s in chain_segments(analyzed.residues)] == [
        ("A", 21), ("B", 30),
    ]


@pytest.mark.network
def test_generate_multimer_document_from_structure():
    result = generate_document(
        _3I40, config=GenerationConfig(max_chains=2), entry_id="3I40"
    )
    assert result is not None
    assert result.num_chains == 2
    assert result.chain_ids == ("A", "B")
    assert result.chain_lengths == (21, 30)
    assert result.seq_len == 51

    tokens = result.document.split()
    assert tokens.count("<n-term>") == 2
    assert tokens.count("<c-term>") == 2

    # Insulin's two chains are disulfide-linked and pack against each other,
    # so the document must carry real interface contacts -- if the
    # sequence-separation filter were applied across the chain boundary most
    # of these would vanish.
    assert result.contacts_emitted_inter_chain > 0
    assert result.contacts_pre_filter_inter_chain >= result.contacts_emitted_inter_chain

    # Every emitted contact's chain pair agrees with the residues it names.
    for contact in result.contacts:
        assert contact.chain_i == result.residues[contact.seq_i].chain
        assert contact.chain_j == result.residues[contact.seq_j].chain
        assert contact.inter_chain == (contact.chain_i != contact.chain_j)


@pytest.mark.network
def test_multimer_inter_chain_contacts_match_pyconfind():
    """The document's interface contacts are exactly pyconfind's, unfiltered.

    Intra-chain contacts get the sequence-separation and minimum-degree
    filters; inter-chain ones only get the degree filter. This checks the
    document against the raw pyconfind analysis rather than against itself.
    """
    config = GenerationConfig(max_chains=2)
    analyzed = analyze_structure(_3I40, max_chains=2)
    chain_of = [r.chain for r in analyzed.residues]
    expected = {
        (c.seq_i, c.seq_j)
        for c in analyzed.contacts
        if chain_of[c.seq_i] != chain_of[c.seq_j]
        and c.degree >= config.min_contact_degree
    }

    result = generate_document(_3I40, config=config, entry_id="3I40")
    assert result is not None
    # Insulin is small enough that nothing is truncated, so every eligible
    # interface contact should be present.
    assert not result.truncated
    emitted = {(c.seq_i, c.seq_j) for c in result.contacts if c.inter_chain}
    assert emitted == expected
    assert expected
