"""Tests for structural-first selection with unconditional quality filling."""

import itertools

import pytest

from production_policy import select_three, select_three_dynamic


def member(entry_id: str, *, anchor: bool = False, plddt: float = 90) -> dict:
    """Make one quality-passing production member."""
    return {
        "entry_id": entry_id,
        "struct_cluster_id": "cluster",
        "is_anchor": anchor,
        "seq_len": 100,
        "global_plddt": plddt,
        "ptm": 0.8,
    }


def pairs(rows: list[dict], similarities: dict[str, float]) -> list[dict]:
    """Make complete pair metrics, using the lower named member's score."""
    result = []
    for a, b in itertools.combinations(rows, 2):
        candidate = a if not a["is_anchor"] else b
        score = similarities.get(candidate["entry_id"], 0.95)
        result.append(
            {
                "entry_a": a["entry_id"],
                "entry_b": b["entry_id"],
                "tm_max": score,
                "core_tm_max": score,
                "coverage_a": 0.95,
                "coverage_b": 0.95,
                "aligned_sequence_identity": 0.4,
            }
        )
    return result


def test_selects_strict_diversity_then_fills_to_three() -> None:
    rows = [
        member("anchor", anchor=True),
        member("diverse", plddt=81),
        member("quality-a", plddt=96),
        member("quality-b", plddt=94),
        member("quality-c", plddt=92),
    ]
    chosen = select_three(
        rows,
        pairs(
            rows,
            {
                "diverse": 0.6,
                "quality-a": 0.95,
                "quality-b": 0.94,
                "quality-c": 0.93,
            },
        ),
    )
    assert [row["entry_id"] for row in chosen] == [
        "diverse",
        "quality-a",
        "quality-b",
    ]
    assert [row["selection_tier"] for row in chosen] == [
        "structural_diversity",
        "quality_fill",
        "quality_fill",
    ]


def test_takes_all_candidates_without_structure_work_when_no_choice() -> None:
    rows = [
        member("anchor", anchor=True),
        member("one", plddt=82),
        member("two", plddt=91),
    ]
    chosen = select_three(rows, [])
    assert [row["entry_id"] for row in chosen] == ["two", "one"]
    assert all(row["selection_tier"] == "quality_fill" for row in chosen)
    assert all(row["max_selected_tm"] is None for row in chosen)


def test_missing_pair_metrics_fail_loudly_when_ranking_is_required() -> None:
    rows = [member("anchor", anchor=True)] + [
        member(f"candidate-{index}") for index in range(4)
    ]
    with pytest.raises(ValueError, match="Missing structural metrics"):
        select_three(rows, [])


def test_dynamic_selection_skips_unused_candidate_pairs() -> None:
    rows = [member("anchor", anchor=True)] + [
        member(f"candidate-{index}", plddt=90 + index) for index in range(4)
    ]
    calls = []

    def compare_pair(a: dict, b: dict) -> dict:
        calls.append((a["entry_id"], b["entry_id"]))
        return {
            "tm_max": 0.95,
            "core_tm_max": 0.95,
            "coverage_a": 0.95,
            "coverage_b": 0.95,
            "aligned_sequence_identity": 0.4,
        }

    chosen, measured = select_three_dynamic(rows, compare_pair=compare_pair)
    assert len(chosen) == 3
    assert all(row["selection_tier"] == "quality_fill" for row in chosen)
    assert len(calls) == len(measured) == 4
