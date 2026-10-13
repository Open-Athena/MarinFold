"""Neff regrouping must preserve measurements, scores and comparison populations."""

from pathlib import Path

import numpy as np
import pandas as pd

from prepare import depth_tier
from prepare_neff import regroup
from theme import NEFF_FIGURES, ORDER

HERE = Path(__file__).resolve().parent


def test_fractional_neff_boundaries_and_unchanged_scores() -> None:
    frame = pd.DataFrame(dict(stem=list("abcd"), tier=["≥1000"] * 4, value=[.1, .2, .3, .4]))
    targets = pd.DataFrame(dict(stem=list("abcd"), msa_neff=[9.999, 10., 99.99, 100.], neff_source_row=range(4)))
    targets["neff_tier"] = depth_tier(targets.msa_neff)
    result = regroup(frame, targets)
    assert result.tier.tolist() == ["<10", "10–99", "10–99", "100–999"]
    assert result.value.tolist() == frame.value.tolist()
    assert result.raw_depth_tier.tolist() == frame.tier.tolist()


def test_neff_alignment_identity_and_complete_coverage() -> None:
    targets = pd.read_csv(HERE / "data/targets.csv")
    measurements = pd.read_csv(HERE / "data/msa_neff.csv")
    joined = targets.merge(measurements, on="stem", validate="one_to_one", suffixes=("", "_measured"))
    assert len(joined) == len(targets) == len(measurements) == 333
    assert (joined.msa_depth == joined.msa_depth_measured).all()
    assert (joined.msa_query_length == joined.msa_query_length_measured).all()
    assert (joined.identity_threshold == .8).all()
    assert np.isfinite(joined.msa_neff).all()
    assert joined.msa_neff.between(1 - 1e-10, joined.msa_depth).all()
    assert (joined.validated_rows == np.minimum(joined.msa_depth, 8)).all()


def test_regrouped_values_round_trip_to_original_scores() -> None:
    rows = pd.read_csv(HERE / "data/neff_figure_rows.csv", float_precision="round_trip")
    for source, group in rows.groupby("analysis_source"):
        original = pd.read_csv(HERE / source, float_precision="round_trip")
        indexed = original.iloc[group.analysis_source_row].reset_index(drop=True)
        actual = group.reset_index(drop=True)
        for key in ("stem", "method", "metric", "value", "source", "source_row"):
            pd.testing.assert_series_equal(actual[key], indexed[key], check_names=False)
        assert (actual.raw_depth_tier == indexed.tier).all()
    assert not rows.duplicated(["figure", "stem", "metric", "method"]).any()
    for base, metrics in NEFF_FIGURES.items():
        subset = rows[(rows.figure == base + "_neff") & (rows.designed == 0)]
        expected_n = 305 if base in ("01_predictors", "02_oracle", "05_folding") else 314
        assert set(subset.method) == set(ORDER[base])
        assert set(subset.metric) == set(metrics)
        assert (subset.groupby(["method", "metric"]).stem.nunique() == expected_n).all()
    assert set(rows.loc[rows.figure == "05_folding_neff", "method"]) & {"marinfold_helico", "marinfold_helico_L5"} == set()


def test_af3_extended_sampling_is_only_the_original_five() -> None:
    original = pd.read_csv(HERE / "data/af3_context_rows.csv", float_precision="round_trip")
    rows = pd.read_csv(HERE / "data/neff_af3_context_rows.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(rows[["context_row", "stem", "method", "tm_score"]],
                                  original[["context_row", "stem", "method", "tm_score"]])
    summary = pd.read_csv(HERE / "data/neff_af3_context_summary.csv")
    low = summary[summary.tier == "<10"].set_index("method")
    assert low.loc["af3", "n"] == 14
    assert low.loc["af3_1000_ptm", "n"] == 5
    assert set(rows.loc[rows.method == "af3_1000_ptm", "stem"]) == {"8ii8_A", "8oxk_A", "8qoh_A", "8ux2_A", "8wrx_A"}
