# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-and-distances-v1 DocumentStructure.

The network-marked tests download the pinned published tokenizer
(``timodonnell/protein-docs-tokenizer@83f597d88e9b``) and assert
byte-equivalence with the locally-constructed one. Skip them with
``pytest -m 'not network'`` if you're offline.

Run:

    uv sync --extra test
    uv run pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pytest

# The experiment dir is not a package; make ``structure`` importable.
_EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_DIR))

from structure import (  # noqa: E402
    AMINO_ACIDS,
    ATOM_NAMES,
    CONTACT_TYPES,
    CONTROL_TOKENS,
    DISTANCE_BINS,
    DISTANCE_MARKER,
    MAX_POSITION,
    PLDDT_BINS,
    UNK_TOKEN,
    ContactsAndDistancesV1,
    get_structure,
)

from marinfold_document_structures import (  # noqa: E402
    DocumentStructure,
    build_tokenizer,
)


# -- Protocol conformance ----------------------------------------------------


def test_get_structure_satisfies_protocol():
    s = get_structure()
    assert isinstance(s, DocumentStructure), (
        f"get_structure() returned a {type(s).__name__} that does not "
        "satisfy the DocumentStructure protocol."
    )


def test_structure_metadata():
    s = get_structure()
    assert s.name == "contacts-and-distances-v1"
    assert s.context_length == 8192


# -- tokens() shape ----------------------------------------------------------


def test_tokens_count_matches_canonical_2838():
    """Domain vocab is 2838 tokens; +2 specials = 2840 total.

    Composition: 4 control + 3 contact-types + 1 distance-marker
    + 64 distance-bins + 7 plddt-bins + 20 AAs + 37 atoms
    + 2701 positions + 1 UNK = 2838.
    """
    tokens = get_structure().tokens()
    n_expected = (
        len(CONTROL_TOKENS)
        + len(CONTACT_TYPES)
        + len(DISTANCE_MARKER)
        + len(DISTANCE_BINS)
        + len(PLDDT_BINS)
        + len(AMINO_ACIDS)
        + len(ATOM_NAMES)
        + (MAX_POSITION + 1)
        + len(UNK_TOKEN)
    )
    assert n_expected == 2838, f"category counts don't sum to 2838: {n_expected}"
    assert len(tokens) == 2838


def test_tokens_unique():
    tokens = get_structure().tokens()
    assert len(tokens) == len(set(tokens)), "tokens() returned duplicates"


def test_tokens_returns_a_copy():
    """Mutating the caller's copy must not affect the cached canonical list."""
    s = ContactsAndDistancesV1()
    a = s.tokens()
    a.append("<broken>")
    b = s.tokens()
    assert "<broken>" not in b


def test_token_order_invariants():
    """Spot-check critical positions in the canonical ordering.

    These positions are baked into every checkpoint trained against
    the v1 vocab. After accounting for the 2 specials (<pad>, <eos>)
    prepended by build_tokenizer, the document-text tokens land at
    these IDs in the published tokenizer.
    """
    tokens = get_structure().tokens()
    # Domain-token list starts at id 2 after specials.
    assert tokens[0] == "<contacts-and-distances-v1>"
    assert tokens[1] == "<begin_sequence>"
    assert tokens[4] == "<long-range-contact>"
    assert tokens[7] == "<distance>"
    assert tokens[8] == "<d0.5>"
    assert tokens[8 + 63] == "<d32.0>"
    assert tokens[-1] == "<UNK>"


# -- build_tokenizer ---------------------------------------------------------


def test_build_tokenizer_size():
    tok = build_tokenizer(get_structure())
    # Domain 2838 + <pad> + <eos> = 2840.
    assert len(tok) == 2840
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1
    assert tok.convert_tokens_to_ids("<contacts-and-distances-v1>") == 2


def test_build_tokenizer_roundtrip_sample():
    """Spot-check 1:1 tokenization on a tiny sample document fragment."""
    tok = build_tokenizer(get_structure())
    sample = (
        "<contacts-and-distances-v1> <begin_sequence> "
        "<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU> "
        "<begin_statements> "
        "<long-range-contact> <p1> <p50> "
        "<distance> <p10> <p45> <CA> <CB> <d4.5> "
        "<plddt_80_85> <end>"
    )
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids)
    # WordLevel + whitespace-split tokenizes 1:1 on space-separated
    # tokens, so the count must match.
    assert len(ids) == len(sample.split()), (
        f"non-1:1 tokenization: {len(ids)} ids for {len(sample.split())} tokens"
    )
    # Every token must be in-vocab (id != UNK id, where UNK == "<UNK>"
    # in our build).
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    assert unk_id not in ids, f"sample contained unknown tokens: {sample}"
    # Round-trip should preserve the text exactly (modulo possible
    # tokenizer-level whitespace normalization that is a no-op here).
    assert decoded.strip().split() == sample.split()


# -- Byte-identity vs the pinned published tokenizer -------------------------
#
# These checks require huggingface_hub + network access. Skipped with
# ``pytest -m 'not network'``. They are how we guarantee that
# tokens() + build_tokenizer reproduces the legacy 2840-vocab artifact
# every existing v1 checkpoint was trained against.

REVISION_PIN = "83f597d88e9b"


@pytest.mark.network
def test_published_tokenizer_vocab_matches():
    pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    published = AutoTokenizer.from_pretrained(
        "timodonnell/protein-docs-tokenizer",
        revision=REVISION_PIN,
    )
    local = build_tokenizer(get_structure())

    pub_vocab = published.get_vocab()
    loc_vocab = local.get_vocab()
    assert pub_vocab.keys() == loc_vocab.keys(), (
        f"vocab token set differs: "
        f"only in published = {sorted(set(pub_vocab) - set(loc_vocab))[:10]}; "
        f"only in local = {sorted(set(loc_vocab) - set(pub_vocab))[:10]}"
    )
    # Token IDs must agree for every shared token.
    mismatches = [
        (t, pub_vocab[t], loc_vocab[t])
        for t in pub_vocab
        if pub_vocab[t] != loc_vocab[t]
    ]
    assert not mismatches, (
        f"token-id mismatches against published tokenizer (first 5): "
        f"{mismatches[:5]}"
    )


@pytest.mark.network
def test_published_tokenizer_encodes_identically():
    """A non-trivial sample tokenizes to identical ID sequences."""
    pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    published = AutoTokenizer.from_pretrained(
        "timodonnell/protein-docs-tokenizer",
        revision=REVISION_PIN,
    )
    local = build_tokenizer(get_structure())

    sample = (
        "<contacts-and-distances-v1> <begin_sequence> "
        "<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU> "
        "<begin_statements> "
        "<long-range-contact> <p1> <p50> "
        "<distance> <p10> <p45> <CA> <CB> <d4.5> "
        "<medium-range-contact> <p3> <p20> "
        "<short-range-contact> <p5> <p12> "
        "<distance> <p2> <p80> <NZ> <O> <d15.0> "
        "<plddt_80_85> <end>"
    )
    assert published.encode(sample, add_special_tokens=False) == local.encode(
        sample, add_special_tokens=False
    )


# -- generate / evaluate placeholders (raise NotImplementedError) ------------


def test_generate_documents_not_implemented():
    s = get_structure()
    with pytest.raises(NotImplementedError):
        list(s.generate_documents(iter([])))


def test_evaluate_not_implemented():
    s = get_structure()
    with pytest.raises(NotImplementedError):
        s.evaluate(model_path="fake", ground_truth_records=iter([]))
