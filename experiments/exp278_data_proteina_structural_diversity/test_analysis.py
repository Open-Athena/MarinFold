"""Protect the scientific interpretation of structural clustering and matching."""

from pathlib import Path

import numpy as np
import pytest

from analyze_screen import (
    CONDITIONED_CLASSES,
    components,
    effective_clusters,
    matched_comparison,
    retention_report,
    sequence_exclusion_reasons,
)


def test_weak_evalue_cannot_override_approved_identity_screen() -> None:
    result = sequence_exclusion_reasons(
        {
            "fident": "0.31",
            "qcov": "0.2",
            "tcov": "0.6",
            "qlen": "400",
            "tlen": "100",
            "evalue": "100",
        }
    )
    assert result["identity_rule"]
    assert not result["evalue_rule"]


def test_missing_decontamination_is_not_treated_as_clean(tmp_path: Path) -> None:
    (tmp_path / "quality.csv").write_text("stem,quality_pass\nexample,True\n")
    with pytest.raises(FileNotFoundError, match="sequence-screen.json"):
        retention_report(tmp_path, tmp_path)


def test_excluded_bridge_cannot_join_retained_clusters() -> None:
    edges = [("alpha", "bridge"), ("bridge", "beta"), ("beta", "beta2")]
    result = components(["alpha", "beta", "beta2"], edges)
    assert {frozenset(group) for group in result} == {
        frozenset(["alpha"]),
        frozenset(["beta", "beta2"]),
    }


def test_entropy_penalizes_repeated_structures() -> None:
    stems = list("abcdef")
    equal_sizes = [("a", "b"), ("c", "d"), ("e", "f")]
    uneven_sizes = [("a", "b"), ("b", "c"), ("c", "d")]
    assert np.isclose(effective_clusters(stems, equal_sizes), 3)
    assert effective_clusters(stems, uneven_sizes) < 3


def test_matched_comparison_does_not_reward_larger_conditioned_pool() -> None:
    quality = []
    for condition, count in [
        ("unconditional", 6),
        ("1.x.x.x", 20),
        ("2.x.x.x", 20),
        ("3.x.x.x", 20),
    ]:
        quality.extend(
            {
                "length": "100",
                "condition": condition,
                "stem": f"{condition}-{i}",
                "quality_pass": "True",
            }
            for i in range(count)
        )
    result = matched_comparison(quality, [])
    assert result["matched_n_per_arm"] == 6
    assert result["ratio_median"] == 1


def test_unavailable_class_is_not_silently_removed_from_balanced_comparison() -> None:
    rows = [
        {
            "length": "100",
            "condition": "unconditional",
            "stem": str(i),
            "quality_pass": "True",
        }
        for i in range(6)
    ]
    result = matched_comparison(rows, [])
    assert result["matched_n_per_arm"] == 0
    assert result["ratio_median"] is None


def matched_rows(assignments: list[tuple[str, int]]) -> list[dict]:
    """Build quality rows from (condition, length) pairs, one stem each."""
    return [
        {
            "length": str(length),
            "condition": condition,
            "stem": f"{condition}-{length}-{index}",
            "quality_pass": "True",
        }
        for index, (condition, length) in enumerate(assignments)
    ]


def test_neighbouring_lengths_still_match_after_binning() -> None:
    rows = matched_rows(
        [("unconditional", 100), ("unconditional", 101), ("unconditional", 102)]
        + [("1.x.x.x", 103), ("2.x.x.x", 104), ("3.x.x.x", 105)]
    )
    assert matched_comparison(rows, [], bin_width=1)["matched_n_per_arm"] == 0
    result = matched_comparison(rows, [])
    assert result["matched_n_per_arm"] == 3
    assert result["bins_contributing"] == 1


def test_distant_lengths_are_not_merged_into_one_bin() -> None:
    rows = matched_rows(
        [("unconditional", 60), ("unconditional", 61), ("unconditional", 62)]
        + [("1.x.x.x", 400), ("2.x.x.x", 401), ("3.x.x.x", 402)]
    )
    assert matched_comparison(rows, [])["matched_n_per_arm"] == 0


def test_saturated_arms_report_the_ratio_ceiling() -> None:
    rows = matched_rows(
        [("unconditional", 100 + i) for i in range(9)]
        + [(condition, 100 + i) for condition in CONDITIONED_CLASSES for i in range(3)]
    )
    result = matched_comparison(rows, [])
    assert result["matched_n_per_arm"] == 9
    assert np.isclose(result["ratio_median"], 1)
    assert np.isclose(result["maximum_possible_ratio_median"], 1)
    assert result["ceiling_limited"] is True
    assert "untestable" in result["interpretation"]


def test_unmatched_arm_reports_unavailable_rather_than_a_null_result() -> None:
    rows = matched_rows([("unconditional", 100) for _ in range(6)])
    result = matched_comparison(rows, [])
    assert result["ceiling_limited"] is None
    assert "not a null result" in result["interpretation"]
