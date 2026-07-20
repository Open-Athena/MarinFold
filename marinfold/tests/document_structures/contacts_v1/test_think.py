# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-v1 ``<think>`` (pause) token path (issue #123).

``GenerationConfig(think=True)`` splices ``<think>`` runs *between*
``<contact>`` statements in the structure section, using the same
distributions as #34's contacts-and-distances-v2. These tests cover the
sampling laws, the placement contract, the budget accounting, and — most
importantly — that ``think=False`` (the default) leaves the generator
byte-identical to the pre-think version, so existing corpora / checkpoints
are unaffected.
"""

import random
import statistics

import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
    generate_sequence_only_document,
    _geometric,
    _generation_seed,
    _sample_think_overhead,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo


_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]


def _residues(n: int) -> list[ResidueInfo]:
    return [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=1 + i, chain="A")
        for i in range(n)
    ]


# min_seq_separation=1 keeps every synthetic contact; the seq-sep filter has
# its own tests in test_generate.py.
_CFG_OFF = GenerationConfig(min_seq_separation=1)
_CFG_ON = GenerationConfig(min_seq_separation=1, think=True)


def _structure_tokens(document: str) -> list[str]:
    toks = document.split()
    return toks[toks.index("<begin_statements>") + 1: toks.index("<end>")]


# ---------------------------------------------------------------------------
# think=False is byte-identical to the pre-think generator
# ---------------------------------------------------------------------------


def test_think_off_is_the_default():
    assert GenerationConfig().think is False


def test_think_off_emits_no_think_tokens():
    res = build_document("AF-OFF", _residues(20),
                         [RawContact(i, i + 6, float(20 - i)) for i in range(14)],
                         config=_CFG_OFF)
    assert "<think>" not in res.document.split()
    assert res.think_tokens == 0
    assert res.metadata_row()["think_tokens"] == 0


def test_think_off_matches_default_config_byte_for_byte():
    """Explicit think=False draws no think RNG, so it equals the default path.

    ``_CFG_OFF`` only flips ``min_seq_separation``; ``think`` stays False. If
    enabling the think fields ever perturbed the RNG stream when the switch is
    off, this (and the pinned examples in test_generate.py) would change.
    """
    residues = _residues(15)
    contacts = [RawContact(i, i + 5, float(15 - i)) for i in range(10)]
    default = build_document("AF-BYTE", residues, contacts,
                             config=GenerationConfig(min_seq_separation=1))
    explicit_off = build_document(
        "AF-BYTE", residues, contacts,
        config=GenerationConfig(min_seq_separation=1, think=False,
                                think_initial_prob=0.9),  # inert while off
    )
    assert explicit_off.document == default.document


# ---------------------------------------------------------------------------
# _geometric
# ---------------------------------------------------------------------------


def test_geometric_support_starts_at_1():
    rng = random.Random(0)
    assert all(_geometric(rng, 0.3) >= 1 for _ in range(200))


def test_geometric_p_one_returns_one():
    rng = random.Random(0)
    assert all(_geometric(rng, 1.0) == 1 for _ in range(10))


def test_geometric_mean_matches_1_over_p():
    rng = random.Random(42)
    samples = [_geometric(rng, 0.13) for _ in range(20_000)]
    assert abs(statistics.mean(samples) - 1.0 / 0.13) < 0.3


@pytest.mark.parametrize("bad_p", [0.0, -0.1, 1.5])
def test_geometric_rejects_invalid_p(bad_p):
    with pytest.raises(ValueError):
        _geometric(random.Random(0), bad_p)


# ---------------------------------------------------------------------------
# _sample_think_overhead distributions (match #34)
# ---------------------------------------------------------------------------


def test_sample_think_overhead_p075_gate():
    cfg = GenerationConfig(think=True)
    rng = random.Random(1)
    n = 5000
    fired = sum(1 for _ in range(n) if _sample_think_overhead(rng, cfg)[0] > 0)
    assert abs(fired / n - cfg.think_initial_prob) < 0.02


def test_sample_think_overhead_k1_geom_p013():
    cfg = GenerationConfig(think=True)
    rng = random.Random(2)
    fired = [k1 for _ in range(20_000) if (k1 := _sample_think_overhead(rng, cfg)[0]) > 0]
    assert abs(statistics.mean(fired) - 1.0 / cfg.think_initial_geom_p) < 0.3


def test_sample_think_overhead_additional_run_count():
    # E[max(int(U(-4, 4)), 0)] = (0+1+2+3)/8 = 0.75; support 0..3 (never 4).
    cfg = GenerationConfig(think=True)
    rng = random.Random(3)
    counts = [len(_sample_think_overhead(rng, cfg)[1]) for _ in range(20_000)]
    assert abs(statistics.mean(counts) - 0.75) < 0.05
    assert all(0 <= c <= 4 for c in counts)


def test_sample_think_overhead_run_length_geom_p025():
    cfg = GenerationConfig(think=True)
    rng = random.Random(4)
    lengths: list[int] = []
    for _ in range(20_000):
        lengths.extend(_sample_think_overhead(rng, cfg)[1])
    assert lengths
    assert abs(statistics.mean(lengths) - 1.0 / cfg.think_run_length_geom_p) < 0.2


# ---------------------------------------------------------------------------
# Document shape with think=True
# ---------------------------------------------------------------------------


def test_think_document_framing_and_tokenizes_without_unk():
    res = build_document("AF-ON", _residues(30),
                         [RawContact(i, i + 6, float(30 - i)) for i in range(20)],
                         config=_CFG_ON)
    toks = res.document.split()
    assert toks[0] == "<contacts-v1>"
    assert toks[1] == "<begin_sequence>"
    assert toks[-1] == "<end>"
    assert res.think_tokens > 0
    assert toks.count("<think>") == res.think_tokens
    tok = build_tokenizer(vocab.all_domain_tokens())
    ids = tok.encode(res.document, add_special_tokens=False)
    assert len(ids) == len(toks)
    assert tok.convert_tokens_to_ids("<UNK>") not in ids


def test_think_deterministic_same_entry_id():
    residues, contacts = _residues(25), [RawContact(i, i + 6, float(25 - i)) for i in range(15)]
    a = build_document("AF-DET", residues, contacts, config=_CFG_ON)
    b = build_document("AF-DET", residues, contacts, config=_CFG_ON)
    assert a.document == b.document


def test_think_golden_document():
    """Pin exact think placement for a fixed input (guards against drift).

    think_initial_prob=1.0 forces the initial run so the example is stable.
    Slot-0 gets the initial run *and* any additional run that lands there
    (concatenated); note a later run sits before a subsequent <contact>.
    """
    residues = _residues(12)
    contacts = [RawContact(0, 6, 0.9), RawContact(1, 8, 0.6), RawContact(3, 11, 0.3)]
    cfg = GenerationConfig(min_seq_separation=1, think=True, think_initial_prob=1.0)
    expected = (
        "<contacts-v1> <begin_sequence> <p1232> <ALA> <p1231> <MET> <p1242> <ALA> "
        "<n-term> <p1231> <p1240> <ILE> <p1234> <LYS> <p1233> <GLY> <p1238> <VAL> "
        "<p1239> <LEU> <c-term> <p1242> <p1237> <THR> <p1236> <SER> <p1241> <MET> "
        "<p1235> <PHE> <begin_statements> <think> <think> <think> <think> <think> "
        "<think> <contact> <p1231> <p1237> <contact> <p1239> <p1232> <think> <think> "
        "<think> <think> <think> <think> <think> <think> <think> <think> <think> "
        "<think> <think> <think> <think> <think> <think> <think> <think> <think> "
        "<contact> <p1234> <p1242> <end>"
    )
    res = build_document("golden-think", residues, contacts, config=cfg)
    assert res.document == expected
    assert res.think_tokens == 26


# ---------------------------------------------------------------------------
# Placement contract
# ---------------------------------------------------------------------------


def test_think_runs_only_between_contacts_never_within():
    """Every maximal <think> run is followed by <contact> or <end> — never a
    position token (which would mean the run split a <contact> <pX> <pY>)."""
    res = build_document("AF-PLACE", _residues(40),
                         [RawContact(i, i + 6, float(40 - i)) for i in range(28)],
                         config=_CFG_ON)
    struct = _structure_tokens(res.document)
    assert struct.count("<think>") == res.think_tokens > 0
    i = 0
    while i < len(struct):
        if struct[i] != "<think>":
            i += 1
            continue
        j = i
        while j < len(struct) and struct[j] == "<think>":
            j += 1
        nxt = struct[j] if j < len(struct) else "<contact>"  # end of struct == before <end>
        assert nxt == "<contact>", f"think run {i}..{j} followed by {nxt!r}"
        i = j


def test_initial_run_lands_right_after_begin_statements_when_gate_fires():
    cfg = GenerationConfig(min_seq_separation=1, think=True, think_initial_prob=1.0)
    res = build_document("AF-SLOT0", _residues(20),
                         [RawContact(i, i + 6, float(20 - i)) for i in range(12)],
                         config=cfg)
    toks = res.document.split()
    bs = toks.index("<begin_statements>")
    assert toks[bs + 1] == "<think>"


def test_total_think_count_matches_sampled_overhead():
    """With statements present, every sampled run is emitted, so the doc's
    <think> count equals the pre-sampled k1 + sum(additional runs)."""
    residues = _residues(40)
    contacts = [RawContact(i, i + 6, float(40 - i)) for i in range(28)]
    cfg = _CFG_ON
    rng = random.Random(_generation_seed("AF-COUNT"))
    k1, additional = _sample_think_overhead(rng, cfg)
    res = build_document("AF-COUNT", residues, contacts, config=cfg)
    assert res.think_tokens == k1 + sum(additional)
    assert res.document.split().count("<think>") == res.think_tokens


# ---------------------------------------------------------------------------
# Budget + edge cases
# ---------------------------------------------------------------------------


def test_think_tokens_subtracted_from_budget():
    """A forced large think overhead crowds out contacts and never overflows."""
    residues = _residues(50)
    contacts = [RawContact(i, i + 6, float(50 - i)) for i in range(40)]
    ctx = 260  # frame(4)+seq(100)+termini(4)=108 → 50 contact-slots without think
    off = build_document("AF-BUD", residues, contacts, context_length=ctx,
                         config=GenerationConfig(min_seq_separation=1))
    on = build_document("AF-BUD", residues, contacts, context_length=ctx,
                        config=GenerationConfig(min_seq_separation=1, think=True,
                                                think_initial_prob=1.0,
                                                think_initial_geom_p=0.05))
    assert on.num_tokens <= ctx
    assert on.think_tokens > 0
    # Reserved think budget leaves room for strictly fewer contacts.
    assert on.contacts_emitted < off.contacts_emitted


def test_no_contacts_still_emits_initial_run():
    """Edge case (mirrors #34): no contacts but the gate fires → the initial
    run is still emitted; additional runs have no slot and are dropped."""
    cfg = GenerationConfig(min_seq_separation=1, think=True, think_initial_prob=1.0)
    res = build_document("AF-EMPTY", _residues(6), [], config=cfg)
    toks = res.document.split()
    assert res.contacts_emitted == 0
    assert "<contact>" not in toks
    assert toks[-1] == "<end>"
    # Only the initial run is emitted (k1), so think_tokens == the slot-0 run.
    rng = random.Random(_generation_seed("AF-EMPTY"))
    k1, _ = _sample_think_overhead(rng, cfg)
    assert res.think_tokens == k1 > 0
    assert toks.count("<think>") == k1
    # And it sits right after <begin_statements>.
    bs = toks.index("<begin_statements>")
    assert toks[bs + 1] == "<think>"


def test_sequence_only_ignores_think():
    """The sequence-only path has no structure section, so think is inert and
    the document stays byte-identical to the non-think sequence-only doc."""
    seq = "MAGKFSTVLIMAGKFS"
    base = generate_sequence_only_document(
        seq, entry_id="AF-SEQ", config=GenerationConfig())
    with_think = generate_sequence_only_document(
        seq, entry_id="AF-SEQ",
        config=GenerationConfig(think=True, think_initial_prob=1.0))
    assert "<think>" not in with_think.document.split()
    assert with_think.think_tokens == 0
    assert with_think.document == base.document
