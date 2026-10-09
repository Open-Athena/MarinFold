"""Check aggregate scoring against small contact maps with known rankings."""

import pytest

from analyze import aggregate_contacts


def test_weighting_demotes_a_repeated_wrong_contact():
    shared = dict(entry_id='p',group_id='g',gt_count=2,predicted_recall=.5)
    rows = [dict(shared,pairs=[[0,8],[1,9]],labels=[1,0],probabilities=[.9,.1]),
        dict(shared,pairs=[[1,9],[2,10]],labels=[0,1],probabilities=[.1,.8])]
    scored = aggregate_contacts(rows)[0]
    assert scored['frequency_r_precision'] == .5
    assert scored['weighted_r_precision'] == 1.
    assert scored['random_rollout_f1'] == .5


def test_r_precision_counts_missing_reference_contacts_as_missed():
    row = dict(entry_id='p',group_id='g',gt_count=3,predicted_recall=.3,
        pairs=[[0,8]],labels=[1],probabilities=[.9])
    assert aggregate_contacts([row])[0]['weighted_r_precision'] == pytest.approx(1/3)
