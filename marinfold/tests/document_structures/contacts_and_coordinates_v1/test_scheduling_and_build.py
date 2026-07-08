# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Mention-scheduling + build_document tests (pure, no pyconfind)."""

import random

import pytest

from marinfold.document_structures.contacts_v1.generate import (
    build_document as contacts_v1_build_document,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_and_coordinates_v1.generate import (
    GenerationConfig,
    _sample_depth,
    build_document,
)


_RESNAMES = ["MET", "ALA", "GLY", "PHE", "LYS", "THR", "VAL", "SER", "TYR", "TRP"]


def _toy_residues(n: int = 10) -> list[ResidueInfo]:
    return [
        ResidueInfo(seq_index=i, resname=_RESNAMES[i % len(_RESNAMES)],
                    resnum=i + 1, chain="A")
        for i in range(n)
    ]


def _toy_atoms(residues, *, seed: int = 0):
    """Give every residue N/CA/CB heavy atoms at spread-out coordinates."""
    rng = random.Random(seed)
    atoms = {}
    for r in residues:
        base = (rng.uniform(0, 60), rng.uniform(0, 60), rng.uniform(0, 60))
        atoms[r.seq_index] = tuple(
            (name, base[0] + dx, base[1] + dy, base[2] + dz)
            for name, dx, dy, dz in (("N", 0, 0, 0), ("CA", 1.2, 0.4, 0.1),
                                     ("CB", 1.5, 1.4, 0.6))
        )
    return atoms


def _coord_statements(document: str):
    """Extract coordinate statements as (pos, atom, [xyz...]) tuples."""
    toks = document.split()
    j = toks.index("<begin_statements>") + 1
    out = []
    while j < len(toks) and toks[j] != "<end>":
        if toks[j] == "<contact>":
            j += 3
            continue
        pos, atom = toks[j], toks[j + 1]
        j += 2
        xyzs = []
        while j < len(toks) and toks[j].startswith("<xyz-"):
            xyzs.append(toks[j])
            j += 1
        out.append((pos, atom, xyzs))
    return out


def _sequence_section(document: str) -> list[str]:
    toks = document.split()
    lo = toks.index("<begin_sequence>") + 1
    hi = toks.index("<begin_statements>")
    return toks[lo:hi]


def test_sample_depth_shallow_at_t0_deep_at_t1():
    rng = random.Random(0)
    config = GenerationConfig()  # max_depth = 3
    n = 20_000
    at0 = [_sample_depth(rng, 0.0, config) for _ in range(n)]
    at1 = [_sample_depth(rng, 1.0, config) for _ in range(n)]
    # ~91% mass on depth 1 at t=0, on depth 3 (the finest) at t=1.
    assert at0.count(1) / n > 0.86
    assert at1.count(3) / n > 0.86
    # No depth is ever above max_depth; epsilon keeps the extremes reachable.
    assert max(at0) <= 3 and max(at1) <= 3
    assert at0.count(3) > 0 and at1.count(1) > 0


def test_first_coordinate_event_is_full_precision():
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    result = build_document("seed-abc", residues, [], atoms, context_length=32768)
    assert result is not None
    statements = _coord_statements(result.document)
    assert statements, "expected coordinate statements"
    # SPEC: the very first coordinate statement always gets depth max_depth (3).
    assert len(statements[0][2]) == 3
    # Every mention has 1..3 xyz tokens.
    assert all(1 <= len(xyzs) <= 3 for _pos, _atom, xyzs in statements)


def test_first_event_forced_flag_off_allows_shallow_first():
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    config = GenerationConfig(force_full_precision_first_event=False)
    result = build_document("seed-abc", residues, [], atoms,
                            context_length=32768, config=config)
    assert result is not None
    # depth-3-first is no longer guaranteed; the depth-1-heavy schedule at
    # t~0 means the first event is very likely shallow. Only assert it's valid.
    first = _coord_statements(result.document)[0]
    assert 1 <= len(first[2]) <= 3


def test_max_depth_4_knob_produces_four_token_first_event():
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    config = GenerationConfig(max_depth=4)
    result = build_document("seed-abc", residues, [], atoms,
                            context_length=32768, config=config)
    assert result is not None
    statements = _coord_statements(result.document)
    assert len(statements[0][2]) == 4  # forced full precision at max_depth=4
    assert all(1 <= len(xyzs) <= 4 for _pos, _atom, xyzs in statements)
    assert result.max_depth == 4
    assert len(result.depth_histogram) == 4


def test_determinism_same_entry_id():
    residues = _toy_residues(12)
    atoms = _toy_atoms(residues)
    contacts = [RawContact(0, 8, 0.5), RawContact(1, 9, 0.4), RawContact(2, 11, 0.3)]
    a = build_document("dup", residues, contacts, atoms, context_length=4096)
    b = build_document("dup", residues, contacts, atoms, context_length=4096)
    assert a is not None and b is not None
    assert a.document == b.document
    assert a.sha1 == b.sha1
    assert a.metadata_row() == b.metadata_row()


def test_sequence_section_byte_identical_to_contacts_v1():
    # Same residues + entry_id: the n-terminal start index and the
    # sequence-section shuffle are the first two RNG draws in both formats,
    # so the sequence section is byte-for-byte the same.
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    ours = build_document("shared-seed", residues, [], atoms, context_length=32768)
    theirs = contacts_v1_build_document("shared-seed", residues, [])
    assert ours is not None and theirs is not None
    assert _sequence_section(ours.document) == _sequence_section(theirs.document)


def test_budget_is_respected_and_truncation_flag():
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    # A tight budget: sequence section is fixed(=4)+2*10+4 = 28 tokens, so a
    # small context forces the coordinate section to truncate.
    result = build_document("trunc", residues, [], atoms, context_length=60)
    assert result is not None
    assert result.num_tokens <= 60
    assert result.num_tokens == len(result.document.split())
    assert result.truncated is True


def test_large_budget_not_truncated_for_small_structure():
    residues = _toy_residues(6)
    atoms = _toy_atoms(residues)
    result = build_document("big", residues, [], atoms, context_length=32768)
    assert result is not None
    assert result.num_tokens <= 32768
    # depth histogram sums to the event count.
    assert sum(result.depth_histogram) == result.num_events


def test_too_few_residues_returns_none():
    residues = _toy_residues(1)
    atoms = _toy_atoms(residues)
    assert build_document("x", residues, [], atoms) is None


def test_contacts_zero_prob_and_cap():
    residues = _toy_residues(60)
    atoms = _toy_atoms(residues)
    contacts = [RawContact(i, i + 6, 0.5) for i in range(50)]
    # With zero prob 1.0, never emit contacts.
    cfg_zero = GenerationConfig(n_contacts_zero_prob=1.0)
    r0 = build_document("cz", residues, contacts, atoms,
                        context_length=32768, config=cfg_zero)
    assert r0 is not None and r0.contacts_emitted == 0
    # With zero prob 0.0 and a small cap, emit between 1 and the cap.
    cfg_cap = GenerationConfig(n_contacts_zero_prob=0.0, n_contacts_max=5)
    r1 = build_document("cz", residues, contacts, atoms,
                        context_length=32768, config=cfg_cap)
    assert r1 is not None and 1 <= r1.contacts_emitted <= 5
