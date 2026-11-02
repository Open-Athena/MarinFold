"""Check aggregate scoring against small contact maps with known rankings."""

import hashlib

import pandas as pd
import pytest

from analyze import aggregate_contacts, validate_evaluation


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


def test_evaluation_rejects_stale_rank_and_missing_protein():
    common = dict(world=2,checkpoint='step-500',split='test',data_fingerprint='frozen-corpus',
        expected_proteins=2,expected_ids_sha256=hashlib.sha256(b'["a", "b"]').hexdigest(),
        rows=1,proteins=1)
    markers = [dict(common,rank=0),dict(common,rank=1)]
    frame = pd.DataFrame([dict(entry_id='a',identity='a:r0',prefix='full'),
        dict(entry_id='b',identity='b:r0',prefix='full')])
    validate_evaluation(markers,frame)
    with pytest.raises(ValueError,match='checkpoint'):
        validate_evaluation([markers[0],dict(markers[1],checkpoint='step-1000')],frame)
    with pytest.raises(ValueError,match='missing completed ranks'):
        validate_evaluation(markers[:1],frame.iloc[:1])
    with pytest.raises(ValueError,match='coverage'):
        validate_evaluation(markers,frame.replace({'entry_id':{'b':'unexpected'}}))
