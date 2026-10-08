"""Audit population matching and confidence selection in the AF3 comparison."""

from pathlib import Path

import numpy as np
import pandas as pd

from prepare_af3_context import select_samples

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def test_every_context_score_round_trips_to_its_original_cell() -> None:
    rows = pd.read_csv(HERE / "data/af3_context_rows.csv", float_precision="round_trip")
    for source, group in rows.groupby("source"):
        original = pd.read_csv(REPO / source, float_precision="round_trip")
        for row in group.itertuples():
            raw = original.iloc[row.source_row]
            assert row.tm_score == raw[row.source_column]
            assert row.stem == raw.get("target_id", raw.stem)


def test_baselines_match_and_extended_sampling_stays_in_lowest_bin() -> None:
    rows = pd.read_csv(HERE / "data/af3_context_rows.csv")
    source = pd.read_csv(HERE / "data/figure_rows.csv")
    expected = set(source.loc[(source.figure == "01_predictors") & (source.designed == 0), "stem"])
    assert len(expected) == 305
    for _, group in rows[rows.series == "archived_predictor"].groupby("method"):
        assert set(group.stem) == expected
        assert len(group) == 305
    extra = rows[rows.series == "extended_af3"]
    assert set(extra.tier) == {"<10"}
    assert extra.stem.nunique() == 5
    assert not rows.duplicated(["stem", "method"]).any()
    summary = pd.read_csv(HERE / "data/af3_context_summary.csv")
    assert summary.loc[summary.method == "af3"].set_index("tier").n.to_dict() == {
        "<10": 5, "10–99": 20, "100–999": 60, "≥1000": 220}


def test_confidence_selection_ignores_accuracy_and_samples_beyond_budget() -> None:
    samples = pd.DataFrame([
        dict(stem="p", seed=12, ptm=0.9, tm_score=0.99, source="test", source_row=0),
        dict(stem="p", seed=10, ptm=0.8, tm_score=0.3, source="test", source_row=1),
        dict(stem="p", seed=11, ptm=0.8, tm_score=0.95, source="test", source_row=2),
    ])
    row = select_samples(samples, 2, "ptm").iloc[0]
    assert row.selected_seed == 10
    assert row.tm_score == 0.3
    assert not row.uses_ground_truth


def test_new_selections_agree_with_independent_prefix_analysis() -> None:
    rows = pd.read_csv(HERE / "data/af3_context_rows.csv")
    old = pd.read_csv(HERE / "data/af3_sampling_summary.csv")
    for row in rows[rows.series == "extended_af3"].itertuples():
        expected = old[(old.stem == row.stem) & (old.budget == row.budget)].iloc[0]
        column = {"ptm": "ptm_selected_tm", "ranking_score": "ranking_selected_tm", "tm_score": "best_tm"}[row.selector]
        assert np.isclose(row.tm_score, expected[column], atol=1e-12, rtol=0)
