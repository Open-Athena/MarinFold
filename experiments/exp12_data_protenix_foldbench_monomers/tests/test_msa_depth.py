# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for msa_depth.py (a3m parsing + N_eff reweighting)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import msa_depth as md  # noqa: E402


def test_parse_a3m_basic():
    text = ">query\nACDEF\n>hit1\nACDEF\n"
    assert md.parse_a3m_sequences(text) == ["ACDEF", "ACDEF"]


def test_parse_a3m_multiline_and_trailing():
    text = ">q\nACD\nEF\n>h\nGHIKL"
    assert md.parse_a3m_sequences(text) == ["ACDEF", "GHIKL"]


def test_match_array_strips_lowercase_inserts():
    # Lowercase letters are insertions relative to the query and dropped;
    # uppercase + '-' are match columns. Second sequence has insertion 'y'.
    seqs = ["ACDEF", "AyCD-F"]
    arr = md.a3m_to_match_array(seqs)
    assert arr.shape == (2, 5)
    # Row 1 match state is "ACD-F" (the 'y' insertion removed).
    assert bytes(arr[1].tolist()).decode() == "ACD-F"


def test_match_array_inconsistent_length_raises():
    with pytest.raises(ValueError):
        md.a3m_to_match_array(["ACDEF", "ACD"])


def test_neff_identical_sequences_is_one():
    arr = md.a3m_to_match_array(["ACDEF"] * 5)
    assert md.compute_neff(arr, threshold=0.8) == pytest.approx(1.0)


def test_neff_fully_diverse_is_n():
    # Three sequences with zero overlap identity at any column.
    arr = md.a3m_to_match_array(["AAAAA", "CCCCC", "DDDDD"])
    assert md.compute_neff(arr, threshold=0.8) == pytest.approx(3.0)


def test_neff_two_clusters():
    # Two identical + one distinct => clusters {2, 1} => n_eff = 1/2+1/2+1 = 2.
    arr = md.a3m_to_match_array(["ACDEF", "ACDEF", "GHIKL"])
    assert md.compute_neff(arr, threshold=0.8) == pytest.approx(2.0)


def test_neff_threshold_sensitivity():
    # 80% identity: differs in 1/5 columns -> identity 0.8 -> neighbors.
    # 0.8 threshold (>=) clusters them; a stricter pass would not.
    arr = md.a3m_to_match_array(["ACDEF", "ACDEG"])  # identity 4/5 = 0.8
    assert md.compute_neff(arr, threshold=0.8) == pytest.approx(1.0)
    # At a threshold above 0.8 they split into two clusters.
    assert md.compute_neff(arr, threshold=0.9) == pytest.approx(2.0)


def test_neff_overlap_normalized_for_partial_coverage():
    # A short fragment that perfectly matches its overlap clusters with
    # the query (identity over both-non-gap columns == 1.0), not distant.
    arr = md.a3m_to_match_array(["ACDEF", "AC---"])
    assert md.compute_neff(arr, threshold=0.8) == pytest.approx(1.0)


def test_neff_blocking_matches_single_block():
    rng = np.random.default_rng(0)
    alphabet = np.frombuffer(b"ACDEFGHIKL", dtype=np.uint8)
    arr = alphabet[rng.integers(0, len(alphabet), size=(40, 12))].astype(np.uint8)
    full = md.compute_neff(arr, threshold=0.62, max_block_elems=10**9)
    chunked = md.compute_neff(arr, threshold=0.62, max_block_elems=50)
    assert full == pytest.approx(chunked)


def test_msa_depth_end_to_end():
    text = ">query\nACDEF\n>h1\nACDEF\n>h2\nGHIKL\n"
    depth = md.msa_depth(text, thresholds=(0.8,))
    assert depth.n_seqs == 3
    assert depth.query_len == 5
    assert depth.n_eff[0.8] == pytest.approx(2.0)
