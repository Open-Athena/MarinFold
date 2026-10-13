"""Independent checks for the curated three-way poster analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export_poster_neff import chosen_names
from prepare_poster_product import ranked_pairs

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def test_ranking_filters_before_selection_and_ignores_truth() -> None:
    pairs = pd.DataFrame(dict(i=[0, 0, 0, 1, 2], j=[3, 24, 25, 26, 27],
        sep=[3, 24, 25, 25, 25], degree=[1, 0.5, 1, 0, 0.5],
        plddt_i=[1, 1, 0.1, 1, 1], plddt_j=[1, 1, 1, 1, 1],
        true_contact=[True, False, True, True, True]))
    result = ranked_pairs(pairs, 24)
    assert result.index.tolist() == [1, 4, 2]
    flipped = pairs.assign(true_contact=~pairs.true_contact)
    assert ranked_pairs(flipped, 24).index.tolist() == result.index.tolist()


@pytest.mark.skipif(not (HERE/'scratch/contact_confidence/scored').exists(),
                    reason='Requires the optional frozen pair archive documented in CONTACT_CONFIDENCE.md')
def test_new_metrics_match_independent_pair_sort_and_truth_counts() -> None:
    metrics = pd.read_csv(DATA/'poster_product_metrics.csv', float_precision='round_trip')
    truth = {row['stem']: row for row in map(json.loads, (HERE/'scratch/contacts/inputs/gt_universe_scored.jsonl').read_text().splitlines())}
    for stem in ['8ii8_A', '7ur7_A']:
        record = truth[stem]
        resolved = set(record['resolved'])
        for method in metrics.method.unique():
            pairs = pd.read_parquet(HERE/'scratch/contact_confidence/scored'/method/stem/'pairs.parquet').to_dict('records')
            for region, minimum in [('all', 6), ('long', 24)]:
                eligible = [p for p in pairs if p['j']-p['i'] >= minimum and p['degree'] > 0]
                eligible.sort(key=lambda p: (-p['plddt_i']*p['plddt_j']*p['degree'], p['i'], p['j']))
                positives = {(i,j) for i,j,d in record['contacts'] if d >= 0.001 and j-i >= minimum and i in resolved and j in resolved}
                for metric, k in [('r_precision', len(positives)), ('p_at_l2', record['L']//2), ('p_at_l5', record['L']//5)]:
                    label = metric + ('_long' if region == 'long' else '')
                    row = metrics[(metrics.stem==stem)&(metrics.method==method)&(metrics.metric==label)].iloc[0]
                    hits = sum((p['i'], p['j']) in positives for p in eligible[:k])
                    assert row.requested_n_top == k
                    assert row.n_correct == hits
                    assert row.value == hits/k


def test_matched_cohorts_controls_and_fixed_shortfall_denominators() -> None:
    rows = pd.read_csv(DATA/'poster_product_figure_rows.csv', float_precision='round_trip')
    old = pd.read_csv(DATA/'neff_figure_rows.csv', float_precision='round_trip')
    for (_, _), group in rows.groupby(['figure', 'metric']):
        assert group.groupby('method').stem.nunique().eq(333).all()
        assert group[group.designed==0].groupby('method').stem.nunique().eq(314).all()
    controls = rows[rows.method.isin(['marinfold','knn'])].merge(old, on=['figure','metric','stem','method'], suffixes=('_new','_old'), validate='one_to_one')
    np.testing.assert_array_equal(controls.value_new, controls.value_old)
    standard = rows[~rows.method.isin(['marinfold','knn'])]
    np.testing.assert_array_equal(standard.n_top, standard.requested_n_top)
    assert standard.source.eq('data/poster_product_metrics.csv').all()
    metrics = pd.read_csv(DATA/'poster_product_metrics.csv', float_precision='round_trip')
    np.testing.assert_allclose(metrics.value, metrics.n_correct/metrics.requested_n_top, rtol=0, atol=1e-12)
    assert (metrics.n_selected <= metrics.requested_n_top).all()
    assert (metrics.n_selected < metrics.requested_n_top).any()


def test_curated_inventory_excludes_raw_depth_and_other_rankings() -> None:
    inventory = json.loads((DATA/'poster_manifest.json').read_text())
    names = chosen_names({row['name'] for row in inventory['figures']})
    assert len(names) == 46
    assert '06_sampling_pl2' in names and '06_sampling_pl2_long' in names
    assert '04_contacts_pl2_neff' in names and '04_contacts_pl2_neff_long' in names
    assert '04_contacts_neff' in names and '04_contacts_neff_long' in names
    assert not any(name.startswith(('04c_', '04d_', '04e_')) for name in names)
    assert '01_predictors' not in names and '01_predictors_neff' in names
