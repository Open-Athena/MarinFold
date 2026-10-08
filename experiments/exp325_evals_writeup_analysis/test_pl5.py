"""P@L/5 must retain its denominator and the same protein/source identities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prepare_pl5 import rollout_precision

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def test_short_rollouts_and_duplicate_statements_cannot_inflate_precision() -> None:
    truth = {(0, 8), (1, 10), (2, 20)}
    assert rollout_precision([(0, 8), (0, 8)], truth, 4) == 0.25
    assert rollout_precision([(3, 25), (0, 8), (1, 10)], truth, 2) == 0.5
    assert rollout_precision([], truth, 4) == 0
    with pytest.raises(ValueError):
        rollout_precision([], truth, 0)


def test_all_figure_values_round_trip_to_source_cells() -> None:
    rows = pd.read_csv(HERE / "data/pl5_figure_rows.csv", float_precision="round_trip")
    for source, group in rows.groupby("source"):
        original = pd.read_csv(REPO / source, float_precision="round_trip")
        for row in group.itertuples():
            raw = original.iloc[row.source_row]
            assert row.stem == raw.stem
            assert row.value == raw[row.source_column]


def test_cutoff_uses_sequence_length_and_population_matches_r_precision() -> None:
    rows = pd.read_csv(HERE / "data/pl5_figure_rows.csv")
    old = pd.read_csv(HERE / "data/figure_rows.csv")
    for figure, counterpart in [("04_contacts_pl5", "04_contacts"), ("06_sampling_pl5", "06_sampling")]:
        for method, group in rows[rows.figure == figure].groupby("method"):
            assert set(group.stem) == set(old.loc[(old.figure == counterpart) & (old.method == method), "stem"])
            assert (group.n_top == np.maximum(1, group.L // 5)).all()
    assert not rows.duplicated(["figure", "stem", "method", "metric"]).any()


def test_oracle_is_reselected_at_pl5_from_the_exact_first_100_pool() -> None:
    individuals = pd.read_csv(HERE / "data/pl5_sampling_individual.csv.gz")
    summaries = pd.read_csv(HERE / "data/pl5_sampling_per_protein.csv")
    assert (individuals.loc[~individuals.valid, "p_at_l5"] == 0).all()
    for row in summaries.itertuples():
        pool = individuals[(individuals.stem == row.stem) & (individuals["range"] == row.range)].sort_values("rollout")
        assert pool.rollout.tolist() == list(range(100))
        assert row.single == pytest.approx(pool.p_at_l5.mean(), abs=1e-12)
        assert row.best100 == pytest.approx(pool.p_at_l5.max(), abs=1e-12)
        assert row.best_rollout == int(pool.iloc[np.argmax(pool.p_at_l5.to_numpy())].rollout)
