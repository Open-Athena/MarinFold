"""P@L/2 and P@L/5 must retain their cutoffs and protein/source identities."""

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


@pytest.mark.parametrize("divisor", [2, 5])
def test_all_figure_values_round_trip_to_source_cells(divisor: int) -> None:
    rows = pd.read_csv(HERE / f"data/pl{divisor}_figure_rows.csv", float_precision="round_trip")
    for source, group in rows.groupby("source"):
        original = pd.read_csv(REPO / source, float_precision="round_trip")
        for row in group.itertuples():
            raw = original.iloc[row.source_row]
            assert row.stem == raw.stem
            assert row.value == raw[row.source_column]
            if row.figure == f"04_contacts_pl{divisor}":
                assert raw.cut == f"L/{divisor}"
                assert raw.n_top == row.n_top
                assert raw["range"] == ("long" if row.metric.endswith("_long") else "all")


@pytest.mark.parametrize("divisor", [2, 5])
def test_cutoff_uses_sequence_length_and_population_matches_r_precision(divisor: int) -> None:
    rows = pd.read_csv(HERE / f"data/pl{divisor}_figure_rows.csv")
    old = pd.read_csv(HERE / "data/figure_rows.csv")
    comparisons = [(f"04_contacts_pl{divisor}", "04_contacts")]
    if divisor == 5:
        comparisons.append(("06_sampling_pl5", "06_sampling"))
    for figure, counterpart in comparisons:
        for method, group in rows[rows.figure == figure].groupby("method"):
            assert set(group.stem) == set(old.loc[(old.figure == counterpart) & (old.method == method), "stem"])
            requested = np.maximum(1, group.L // divisor)
            expected = np.minimum(requested, group.n_candidate) if figure.startswith("04_contacts") else requested
            assert (group.n_top == expected).all()
            assert (group.loc[group.designed == 0, "n_top"] == requested[group.designed == 0]).all()
    assert not rows.duplicated(["figure", "stem", "method", "metric"]).any()


@pytest.mark.parametrize("divisor", [2, 5])
def test_oracle_is_reselected_from_the_exact_first_100_pool(divisor: int) -> None:
    individuals = pd.read_csv(HERE / f"data/pl{divisor}_sampling_individual.csv.gz")
    summaries = pd.read_csv(HERE / f"data/pl{divisor}_sampling_per_protein.csv")
    metric = f"p_at_l{divisor}"
    assert (individuals.loc[~individuals.valid, metric] == 0).all()
    assert (individuals.n_top == np.maximum(1, individuals.L // divisor)).all()
    for row in summaries.itertuples():
        pool = individuals[(individuals.stem == row.stem) & (individuals["range"] == row.range)].sort_values("rollout")
        assert pool.rollout.tolist() == list(range(100))
        assert row.single == pytest.approx(pool[metric].mean(), abs=1e-12)
        assert row.best100 == pytest.approx(pool[metric].max(), abs=1e-12)
        assert row.best_rollout == int(pool.iloc[np.argmax(pool[metric].to_numpy())].rollout)


def test_l2_sampling_preserves_pool_identity_and_uses_neff() -> None:
    rows = pd.read_csv(HERE / "data/pl2_sampling_figure_rows.csv", float_precision="round_trip")
    old = pd.read_csv(HERE / "data/pl5_sampling_per_protein.csv")
    scores = pd.read_csv(HERE / "data/pl2_sampling_per_protein.csv", float_precision="round_trip")
    pairs = scores.merge(old, on=["stem", "range"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(pairs) == 314 * 2
    assert pairs.raw_sha256_new.equals(pairs.raw_sha256_old)
    assert pairs.invalid_rollouts_new.equals(pairs.invalid_rollouts_old)
    targets = pd.read_csv(HERE / "data/neff_targets.csv").set_index("stem")
    assert set(rows.stem) == set(old.stem)
    for row in rows.itertuples():
        assert row.tier == targets.loc[row.stem, "neff_tier"]
        source = scores.iloc[row.source_row]
        assert row.stem == source.stem
        assert row.value == source[row.source_column]
