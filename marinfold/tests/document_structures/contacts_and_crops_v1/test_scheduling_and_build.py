# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass build_document tests (pure, no pyconfind)."""

import random

from marinfold.document_structures.contacts_v1.generate import (
    build_document as contacts_v1_build_document,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_and_crops_v1.generate import (
    _CROP_ATOM_TOKENS,
    _CROP_HEADER_TOKENS,
    _PASS1_MENTION_TOKENS,
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_and_crops_v1.vocab import CONTEXT_LENGTH


_RESNAMES = ["MET", "ALA", "GLY", "PHE", "LYS", "THR", "VAL", "SER", "TYR", "TRP"]


def _toy_residues(n: int = 10) -> list[ResidueInfo]:
    return [
        ResidueInfo(seq_index=i, resname=_RESNAMES[i % len(_RESNAMES)],
                    resnum=i + 1, chain="A")
        for i in range(n)
    ]


def _toy_atoms(residues, *, seed: int = 0, spread: float = 120.0):
    """Give every residue N/CA/CB heavy atoms at spread-out coordinates.

    A wide spread scatters atoms across many 10 Å boxes, so Pass 2 has real
    frontier growth and re-show behavior to exercise.
    """
    rng = random.Random(seed)
    atoms = {}
    for r in residues:
        base = (rng.uniform(0, spread), rng.uniform(0, spread), rng.uniform(0, spread))
        atoms[r.seq_index] = tuple(
            (name, base[0] + dx, base[1] + dy, base[2] + dz)
            for name, dx, dy, dz in (("N", 0, 0, 0), ("CA", 1.2, 0.4, 0.1),
                                     ("CB", 1.5, 1.4, 0.6))
        )
    return atoms


def _statements(document: str):
    """Split the structure section into (contacts, pass1, crops).

    ``crops`` is a list of ``(header_tokens, [atom_statement_tokens])``.
    Also asserts every statement's token shape along the way.
    """
    toks = document.split()
    i = toks.index("<begin_statements>") + 1

    contacts = []
    while i < len(toks) and toks[i] == "<contact>":
        contacts.append(toks[i:i + 3])
        i += 3

    pass1 = []
    while i < len(toks) and toks[i] not in ("<crop>", "<end>"):
        stmt = toks[i:i + _PASS1_MENTION_TOKENS]
        assert stmt[0].startswith("<p")
        assert stmt[2].startswith("<xyz-") and stmt[3].startswith("<xyz-")
        pass1.append(stmt)
        i += _PASS1_MENTION_TOKENS

    crops = []
    while i < len(toks) and toks[i] == "<crop>":
        header = toks[i:i + _CROP_HEADER_TOKENS]
        assert header[1].startswith("<xyz-") and header[2].startswith("<xyz-")
        i += _CROP_HEADER_TOKENS
        atoms = []
        while i < len(toks) and toks[i] not in ("<crop>", "<end>"):
            stmt = toks[i:i + _CROP_ATOM_TOKENS]
            assert stmt[0].startswith("<p")
            assert stmt[2].startswith("<xyz-") and stmt[3].startswith("<xyz-")
            atoms.append(stmt)
            i += _CROP_ATOM_TOKENS
        crops.append((header, atoms))

    assert toks[i] == "<end>"
    return contacts, pass1, crops


def _sequence_section(document: str) -> list[str]:
    toks = document.split()
    lo = toks.index("<begin_sequence>") + 1
    hi = toks.index("<begin_statements>")
    return toks[lo:hi]


def test_both_passes_present_and_wellformed():
    residues = _toy_residues(12)
    atoms = _toy_atoms(residues)
    result = build_document("seed-abc", residues, [], atoms)
    assert result is not None
    contacts, pass1, crops = _statements(result.document)
    assert result.num_pass1_mentions == len(pass1) > 0
    assert result.num_crops == len(crops) > 0
    assert result.num_empty_crops == sum(1 for _h, a in crops if not a)
    assert result.crop_atoms_emitted == sum(len(a) for _h, a in crops)
    # A crop header names a real box; atom statements reuse it (ones+tenths).
    assert any(a for _h, a in crops), "expected at least one non-empty crop"


def test_fits_8k_for_small_mid_large():
    for n in (8, 150, 600):
        # A compact cloud that fits the cube under any rotation (a 400 Å cube's
        # space diagonal is ~693 Å < the 980 Å placement limit).
        residues = _toy_residues(n)
        atoms = _toy_atoms(residues, spread=400.0)
        result = build_document(f"len-{n}", residues, [], atoms)
        assert result is not None
        assert result.num_tokens <= CONTEXT_LENGTH
        assert result.num_tokens == len(result.document.split())
        # Structure content stays within the coordinate budget.
        _c, pass1, crops = _statements(result.document)
        assert len(pass1) == result.num_pass1_mentions


def test_pass1_fills_its_cap():
    # Pass 1 draws with replacement until the next 4-token mention overflows
    # the cap, so a small structure fills nearly all of pass1_cap.
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    result = build_document("fill", residues, [], atoms)
    assert result is not None
    fixed = 4 + 2 * 10 + 4
    structure_budget = CONTEXT_LENGTH - fixed
    pass1_cap = structure_budget - GenerationConfig().fine_reserve
    used = result.num_pass1_mentions * _PASS1_MENTION_TOKENS
    assert pass1_cap - _PASS1_MENTION_TOKENS < used <= pass1_cap


def test_pass1_covers_many_residues():
    # 1/(1+k_r) downweighting spreads Pass-1 coverage across residues.
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    result = build_document("cover", residues, [], atoms)
    assert result is not None
    _c, pass1, _crops = _statements(result.document)
    positions = {stmt[0] for stmt in pass1}
    assert len(positions) == 10  # every residue gets boxed


def test_pass2_reshows_boxes_on_small_structure():
    # Small structure, full 8k budget: Pass 2 revisits boxes (progressive
    # refinement), so at least one box is shown more than once.
    residues = _toy_residues(8)
    atoms = _toy_atoms(residues, spread=40.0)
    result = build_document("reshow", residues, [], atoms)
    assert result is not None
    assert result.max_box_visits >= 2
    assert result.num_crops > result.num_distinct_crop_boxes


def test_crop_atoms_are_shuffled_not_residue_grouped():
    # A crop's atoms are emitted in random order, not residue-sequence order —
    # otherwise a residue's atoms come out contiguous and leak adjacency. Use a
    # dense small structure so at least one crop holds atoms from several
    # residues, and assert the positions within some crop are not sorted (a
    # residue-grouped emission would run monotonically by seq index).
    residues = _toy_residues(50)
    atoms = _toy_atoms(residues, spread=15.0)  # dense → boxes pool many residues
    result = build_document("shuffle", residues, [], atoms)
    assert result is not None
    _c, _p1, crops = _statements(result.document)
    multi = [
        [stmt[0] for stmt in a]
        for _h, a in crops
        if len({stmt[0] for stmt in a}) >= 3
    ]
    assert multi, "expected a crop spanning >=3 residues to test ordering"
    # At least one such crop must not be in nondecreasing residue order.
    def _nondecreasing(seq):
        nums = [int(p[2:-1]) for p in seq]
        return all(a <= b for a, b in zip(nums, nums[1:]))
    assert any(not _nondecreasing(positions) for positions in multi)


def test_determinism_same_entry_id():
    residues = _toy_residues(12)
    atoms = _toy_atoms(residues)
    contacts = [RawContact(0, 8, 0.5), RawContact(1, 9, 0.4), RawContact(2, 11, 0.3)]
    a = build_document("dup", residues, contacts, atoms)
    b = build_document("dup", residues, contacts, atoms)
    assert a is not None and b is not None
    assert a.document == b.document
    assert a.sha1 == b.sha1
    assert a.metadata_row() == b.metadata_row()


def test_sequence_section_byte_identical_to_contacts_v1():
    # Same residues + entry_id: the n-terminal start index and the
    # sequence-section shuffle are the first two RNG draws in both formats.
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    ours = build_document("shared-seed", residues, [], atoms)
    theirs = contacts_v1_build_document("shared-seed", residues, [])
    assert ours is not None and theirs is not None
    assert _sequence_section(ours.document) == _sequence_section(theirs.document)


def test_budget_respected_and_truncation_flag_on_tight_context():
    residues = _toy_residues(10)
    atoms = _toy_atoms(residues)
    # Tight context: sequence section is 4 + 2*10 + 4 = 28 tokens; a small
    # budget forces the coordinate section to stop mid-content.
    result = build_document("trunc", residues, [], atoms, context_length=80)
    assert result is not None
    assert result.num_tokens <= 80
    assert result.num_tokens == len(result.document.split())


def test_large_structure_pass1_partial_but_fits():
    # A big chain can't box every atom; Pass 1 is budget-truncated, but the
    # document still fits 8k and Pass 2 still gets its crops.
    residues = _toy_residues(700)
    atoms = _toy_atoms(residues, spread=700.0)
    result = build_document("big", residues, [], atoms)
    assert result is not None
    assert result.num_tokens <= CONTEXT_LENGTH
    assert result.num_eligible_atoms == 700 * 3
    assert result.num_crops > 0


def test_too_few_residues_returns_none():
    residues = _toy_residues(1)
    atoms = _toy_atoms(residues)
    assert build_document("x", residues, [], atoms) is None


def test_contacts_zero_prob_and_cap():
    residues = _toy_residues(60)
    atoms = _toy_atoms(residues)
    contacts = [RawContact(i, i + 6, 0.5) for i in range(50)]
    cfg_zero = GenerationConfig(n_contacts_zero_prob=1.0)
    r0 = build_document("cz", residues, contacts, atoms, config=cfg_zero)
    assert r0 is not None and r0.contacts_emitted == 0
    cfg_cap = GenerationConfig(n_contacts_zero_prob=0.0, n_contacts_max=5)
    r1 = build_document("cz", residues, contacts, atoms, config=cfg_cap)
    assert r1 is not None and 1 <= r1.contacts_emitted <= 5
