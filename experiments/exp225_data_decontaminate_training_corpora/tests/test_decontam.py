# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-logic tests for the exp225 decontamination pass — no mmseqs, no network.

The three things worth testing here are the three places a bug would silently
*under*-filter the corpus rather than fail loudly: inverting a hit back to a
corpus row, deciding whether an alignment is contamination, and resolving a
Foldseek query name back to an eval protein.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decontam_lib import (  # noqa: E402
    ARM_AFDB,
    ARM_ESM,
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    TIER_A,
    TIER_B,
    TIER_C,
    is_sequence_contaminant,
    parse_target,
    tiers_up_to,
)
from structure_droplist import normalize_query  # noqa: E402


class TestParseTarget:
    """exp213's ``{arm}|{shard}_{row}_{entry_id}`` header, inverted."""

    def test_afdb(self):
        row = parse_target("afdb|00000_0_AF-A0A7C3LD06-F1")
        assert (row.arm, row.shard, row.row) == (ARM_AFDB, 0, 0)
        assert row.entry_id == "AF-A0A7C3LD06-F1"
        assert row.key == "afdb|AF-A0A7C3LD06-F1"

    def test_esm_atlas(self):
        row = parse_target("esm_atlas|03337_66759_0000052aa00ab212061f7c6987fd87ae")
        assert (row.arm, row.shard, row.row) == (ARM_ESM, 3337, 66759)
        assert row.entry_id == "0000052aa00ab212061f7c6987fd87ae"

    def test_entry_id_may_contain_underscores(self):
        """Only the first two underscores are structural; the rest are the id."""
        assert parse_target("afdb|00012_7_AF_WEIRD_ID").entry_id == "AF_WEIRD_ID"

    @pytest.mark.parametrize(
        "bad",
        [
            "unknown_arm|00000_0_X",   # arm not in the registry
            "afdb00000_0_X",           # no arm separator
            "afdb|abcde_0_X",          # shard not an integer
            "afdb|00000_x_X",          # row not an integer
            "afdb|00000_0_",           # empty entry_id
            "afdb|00000",              # truncated
        ],
    )
    def test_malformed_raises(self, bad):
        with pytest.raises(ValueError):
            parse_target(bad)


class TestSequenceRule:
    """Tier A is a disjunction; both arms have to fire independently."""

    def test_significant_evalue_alone_is_enough(self):
        # Remote homolog: below the identity bar and below the coverage bar,
        # but significant. This is the case a 40 %-identity funnel misses.
        assert is_sequence_contaminant(identity=0.18, qcov=0.20, evalue=1e-9)

    def test_identity_and_coverage_alone_are_enough(self):
        assert is_sequence_contaminant(identity=0.35, qcov=0.60, evalue=0.5)

    def test_identity_without_coverage_is_not(self):
        # A 95 %-identical fragment over a tenth of the query is not homology.
        assert not is_sequence_contaminant(identity=0.95, qcov=0.10, evalue=0.5)

    def test_coverage_without_identity_is_not(self):
        assert not is_sequence_contaminant(identity=0.25, qcov=0.99, evalue=0.5)

    def test_thresholds_are_inclusive(self):
        assert is_sequence_contaminant(SEQ_MIN_IDENTITY, SEQ_MIN_QCOV, evalue=1.0)
        assert is_sequence_contaminant(identity=0.0, qcov=0.0, evalue=SEQ_MAX_EVALUE)

    def test_the_30_to_40_band_is_what_changed(self):
        """#91's funnel dropped at 40 %; the whole point of 30 % is this band."""
        assert is_sequence_contaminant(identity=0.34, qcov=0.55, evalue=0.4)


class TestTierLadder:
    def test_cumulative(self):
        assert tiers_up_to(TIER_A) == (TIER_A,)
        assert tiers_up_to(TIER_B) == (TIER_A, TIER_B)
        assert tiers_up_to(TIER_C) == (TIER_A, TIER_B, TIER_C)

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError):
            tiers_up_to("D")


class TestNormalizeQuery:
    """Reference keys end in ``_<chain>`` often enough to break naive stripping."""

    KNOWN = {"foldbench100__5sbj_A", "denovo_pdb__1mj0", "cameo_hard__8jvx_B"}

    def test_plain_name_with_extension(self):
        assert normalize_query("foldbench100__5sbj_A.cif", self.KNOWN) == "foldbench100__5sbj_A"

    def test_key_that_itself_ends_in_a_chain_is_not_over_stripped(self):
        # The naive "strip one trailing _<token>" would return
        # "foldbench100__5sbj", which is not a key — and the protein would
        # vanish from the drop list.
        assert normalize_query("foldbench100__5sbj_A", self.KNOWN) == "foldbench100__5sbj_A"

    def test_foldseek_chain_suffix_is_stripped(self):
        assert normalize_query("denovo_pdb__1mj0.cif_A", self.KNOWN) == "denovo_pdb__1mj0"

    def test_unresolvable_raises(self):
        with pytest.raises(ValueError):
            normalize_query("something__else.cif", self.KNOWN)
