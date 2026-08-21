# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-logic tests for the exp225 decontamination pass — no mmseqs, no network.

What is tested here are the places a bug would silently *under*-filter the
corpus rather than fail loudly: inverting a hit back to a corpus row, deciding
whether an alignment is contamination, combining two references, and resolving
a Foldseek query name back to an eval protein.
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
from identity_droplist import dropped_keys  # noqa: E402
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


class TestIdentityRule:
    """The pure identity rule, and that unioning references is set union.

    Reducing two references separately and unioning is only valid because a
    drop is a property of the training row, not of the pairing — the live check
    is that `A + B` equals the reduction of the concatenated alignments, which
    this reproduces in miniature.
    """

    @staticmethod
    def _m8(path, rows):
        # Columns are sequence_droplist.FIELDS:
        # query, target, fident, alnlen, qcov, tcov, evalue, bits
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join("\t".join(str(v) for v in row) + "\n" for row in rows))
        return path

    #: query = eval protein, target = training protein. The three rows differ
    #: only in which side the alignment covers, which is the whole point of
    #: `coverage_mode`.
    COVERAGE_ROWS = (
        # A 12-residue match into a long training protein: covers neither side.
        ("q1", "afdb|00000_0_FRAGMENT", 0.95, 12, 0.12, 0.012, 1.0, 30),
        # Covers most of the eval protein, a sliver of a long training protein.
        ("q1", "afdb|00000_1_LONG_TRAINING", 0.35, 80, 0.80, 0.090, 1.0, 60),
        # A short training protein aligning to one domain of a long eval
        # protein — invisible to a query-side gate, caught by `shorter`.
        ("q2", "afdb|00000_2_SHORT_TRAINING", 0.35, 80, 0.10, 0.850, 1.0, 60),
    )

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("shorter", {"LONG_TRAINING", "SHORT_TRAINING"}),
            ("reference", {"LONG_TRAINING"}),
            ("training", {"SHORT_TRAINING"}),
            ("both", set()),
        ],
    )
    def test_coverage_mode_decides_which_side_must_be_covered(self, tmp_path, mode, expected):
        m8 = self._m8(tmp_path / "aln.m8", self.COVERAGE_ROWS)
        dropped = dropped_keys(
            m8, min_identity=0.30, min_qcov=0.50, max_evalue=None, coverage_mode=mode
        )
        assert dropped[ARM_AFDB] == expected

    def test_a_high_identity_fragment_is_never_homology(self, tmp_path):
        """95 % identical over 12 residues, covering neither sequence."""
        m8 = self._m8(tmp_path / "aln.m8", self.COVERAGE_ROWS[:1])
        for mode in ("shorter", "reference", "training", "both"):
            dropped = dropped_keys(
                m8, min_identity=0.30, min_qcov=0.50, max_evalue=None, coverage_mode=mode
            )
            assert dropped[ARM_AFDB] == set(), mode

    def test_the_remote_arm_is_separable(self, tmp_path):
        """Far below any identity bar, but unmistakably significant."""
        m8 = self._m8(
            tmp_path / "aln.m8",
            [("q1", "afdb|00000_3_REMOTE", 0.10, 90, 0.80, 0.90, 1e-9, 90)],
        )
        kwargs = {"min_identity": 0.30, "min_qcov": 0.50, "coverage_mode": "shorter"}
        assert dropped_keys(m8, max_evalue=None, **kwargs)[ARM_AFDB] == set()
        assert dropped_keys(m8, max_evalue=1e-3, **kwargs)[ARM_AFDB] == {"REMOTE"}

    def test_union_is_set_union_not_a_sum(self, tmp_path):
        """A row homologous to both references must be counted once, not twice."""
        shared = ("qX", "esm_atlas|00001_5_SHARED", 0.60, 90, 0.80, 0.9, 1e-30, 200)
        only_a = ("qA", "esm_atlas|00001_6_A", 0.60, 90, 0.80, 0.9, 1e-30, 200)
        only_b = ("qB", "esm_atlas|00001_7_B", 0.60, 90, 0.80, 0.9, 1e-30, 200)
        a = self._m8(tmp_path / "a" / "aln.m8", [shared, only_a])
        b = self._m8(tmp_path / "b" / "aln.m8", [shared, only_b])
        both = self._m8(tmp_path / "ab" / "aln.m8", [shared, only_a, shared, only_b])

        kwargs = {"min_identity": 0.30, "min_qcov": 0.50, "max_evalue": None}
        left = dropped_keys(a, **kwargs)[ARM_ESM]
        right = dropped_keys(b, **kwargs)[ARM_ESM]
        assert left | right == dropped_keys(both, **kwargs)[ARM_ESM] == {"SHARED", "A", "B"}
        assert len(left | right) < len(left) + len(right)


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
