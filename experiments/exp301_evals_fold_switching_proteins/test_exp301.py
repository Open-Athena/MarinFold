# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exp301 pieces a wrong answer would hide in.

Everything here is pure logic — no GPU, no network. The parts that carry the
conclusions are the coordinate frame (``union_reference``), the contact remap,
the region locator and the rollout parser; a silent error in any of them would
show up as a fold preference rather than as a crash.

    uv run pytest test_exp301.py -q
"""

from __future__ import annotations

from prepare_inputs import (
    locate_fs_region,
    remap_contacts,
    union_reference,
)
from score_foldswitch_worker_cw import canonical, in_region, parse_rollout


class _Contact:
    """Stand-in for pyconfind's contact record."""

    def __init__(self, seq_i: int, seq_j: int, degree: float) -> None:
        self.seq_i, self.seq_j, self.degree = seq_i, seq_j, degree


class _Analyzed:
    def __init__(self, contacts) -> None:
        self.contacts = contacts


# --------------------------------------------------------------------------
# union_reference
# --------------------------------------------------------------------------
def test_identical_chains_give_an_identity_frame():
    obs = "ACDEFGHIKLMNPQRSTVWY"
    frame = union_reference(obs, obs)
    assert frame.reference == obs
    assert frame.map1 == list(range(len(obs)))
    assert frame.map2 == list(range(len(obs)))
    assert frame.diffs == []


def test_reference_is_the_union_of_both_chains_residues():
    # fold1 resolves an N-terminal run the other lacks; fold2 a C-terminal one.
    obs1 = "AAAACDEFGHIK"
    obs2 = "ACDEFGHIKWWW"
    frame = union_reference(obs1, obs2)
    assert frame.reference == "AAAACDEFGHIKWWW"
    # Every mapped residue of each chain must land on its own letter.
    for obs, mapping in ((obs1, frame.map1), (obs2, frame.map2)):
        for k, pos in enumerate(mapping):
            if pos is not None:
                assert frame.reference[pos] == obs[k]


def test_mismatches_are_recorded_and_fold1_supplies_the_reference():
    obs1 = "ACDEFGHIKL"
    obs2 = "ACDEWWHIKL"
    frame = union_reference(obs1, obs2)
    assert frame.reference == obs1
    assert len(frame.diffs) == 1
    diff = frame.diffs[0]
    assert diff["fold1"] == "FG" and diff["fold2"] == "WW"
    assert frame.reference[diff["ref_start"]:diff["ref_start"] + 2] == "FG"


def test_both_mappings_come_from_one_alignment():
    # The bug this guards against: re-deriving fold2's mapping against the
    # finished reference instead of carrying it out of the alignment that built
    # the reference. difflib is greedy, so a second pass need not agree — and
    # when it did not, four pairs (KaiB among them) looked broken.
    obs1 = "ACDEFGHIKLMNPQRSTVWY" + "MMMM"
    obs2 = "ACDEFGHIKL" + "PPPP" + "MNPQRSTVWY"
    frame = union_reference(obs1, obs2)
    mapped2 = [p for p in frame.map2 if p is not None]
    assert mapped2 == sorted(mapped2), "fold2's mapping must be monotonic in the reference"
    for k, pos in enumerate(frame.map2):
        if pos is not None:
            assert frame.reference[pos] == obs2[k]


# --------------------------------------------------------------------------
# remap_contacts
# --------------------------------------------------------------------------
def test_remap_applies_the_degree_and_separation_cuts():
    mapping = list(range(30))
    analyzed = _Analyzed([
        _Contact(0, 10, 0.5),     # kept
        _Contact(0, 3, 0.5),      # separation 3 < 6 -> dropped
        _Contact(1, 20, 1e-6),    # degree below 0.001 -> dropped
        _Contact(25, 5, 0.2),     # kept, and normalised to (5, 25)
    ])
    contacts, resolved = remap_contacts(analyzed, mapping)
    assert contacts == {(0, 10), (5, 25)}
    assert resolved == set(range(30))


def test_remap_drops_contacts_on_unmapped_residues():
    mapping = [0, 1, None, 3]
    analyzed = _Analyzed([_Contact(0, 3, 0.5), _Contact(2, 3, 0.5)])
    contacts, resolved = remap_contacts(analyzed, mapping)
    assert contacts == set()          # separation 3 < 6 kills the first
    assert resolved == {0, 1, 3}      # the None never becomes a position


def test_remap_moves_indices_into_reference_coordinates():
    # fold2's residues sit 4 later in the union reference.
    mapping = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    analyzed = _Analyzed([_Contact(0, 9, 0.5)])
    contacts, _ = remap_contacts(analyzed, mapping)
    assert contacts == {(4, 13)}


# --------------------------------------------------------------------------
# locate_fs_region
# --------------------------------------------------------------------------
def test_exact_region_match():
    reference = "AAAA" + "CDEFGHIKLM" + "TTTT"
    assert locate_fs_region(reference, "CDEFGHIKLM") == (4, 14)


def test_region_split_by_an_unresolved_loop_is_still_found():
    # The table's region sequence carries residues the structure never resolved,
    # so the reference holds it in two blocks. This is the common case (it cost
    # 8 pairs before the locator handled it).
    region = "CDEFGHIKLMNPQRSTVWYA"
    reference = "AAAA" + "CDEFGHIKLM" + "RSTVWYA" + "TTTT"   # "NPQ" missing
    found = locate_fs_region(reference, region)
    assert found is not None
    lo, hi = found
    assert lo == 4 and hi == 21


def test_a_chance_scatter_of_short_matches_is_rejected():
    # Matches that sprawl across the whole chain must not be accepted as a
    # 20-residue region; a wrong location would silently mislabel every
    # FS-restricted count.
    reference = "A" * 200 + "C" * 200
    assert locate_fs_region(reference, "ACACACACACACACACACAC") is None


def test_absent_region_returns_none():
    assert locate_fs_region("A" * 100, "") is None
    assert locate_fs_region("AAAAAAAAAA", "WWWWWWWWWWWWWWWWWWWW") is None


# --------------------------------------------------------------------------
# worker-side parsing and set logic
# --------------------------------------------------------------------------
def test_parse_rollout_maps_wrapped_position_tokens():
    # contacts-v1 starts indexing at a random position and wraps, so the chain's
    # residue 0 here is <p1998>, residue 2 is <p0>, residue 5 is <p3> and
    # residue 10 is <p8>. Getting the wrap wrong would quietly shift every
    # predicted contact and show up as a fold preference.
    seq_index = {(1998 + t) % 2000: t for t in range(20)}
    parsed = parse_rollout(
        "<contact> <p1998> <p8> "     # residues (0, 10), separation 10 -> kept
        "<contact> <p1998> <p0> "     # residues (0, 2),  separation 2  -> dropped
        "<contact> <p3> <p1998>",     # residues (5, 0),  separation 5  -> dropped
        seq_index,
    )
    assert parsed == {(0, 10)}


def test_parse_rollout_is_order_and_duplicate_insensitive():
    seq_index = {t: t for t in range(50)}
    a = parse_rollout("<contact> <p0> <p10> <contact> <p10> <p0>", seq_index)
    b = parse_rollout("<contact> <p10> <p0>", seq_index)
    assert a == b == {(0, 10)}


def test_parse_rollout_ignores_unknown_positions():
    seq_index = {t: t for t in range(20)}
    assert parse_rollout("<contact> <p0> <p1500>", seq_index) == set()


def test_in_region_needs_only_one_endpoint():
    assert in_region((5, 100), (0, 10))
    assert in_region((100, 5), (0, 10))
    assert not in_region((50, 100), (0, 10))
    assert not in_region((5, 100), None)


def test_canonical_reads_the_universe_list_form():
    assert canonical([[3, 9], [1, 8]]) == {(3, 9), (1, 8)}
    assert canonical([]) == set()
