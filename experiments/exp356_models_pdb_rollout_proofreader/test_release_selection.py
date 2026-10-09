"""Check that checkpoint selection cannot favor proteins with more valid rollouts."""

import pandas as pd
import pytest

from select_release import PREFIXES, candidate_objectives


def test_selection_weights_proteins_equally_and_uses_recall_error():
    rows = []
    for prefix in PREFIXES:
        rows.extend([dict(prefix=prefix,entry_id='many',bce=0.,predicted_recall=0.,recall=0.)]*10)
        rows.append(dict(prefix=prefix,entry_id='one',bce=.75,predicted_recall=.5,recall=0.))
    assert list(candidate_objectives(pd.DataFrame(rows)).values()) == [.5]*len(PREFIXES)


def test_selection_rejects_nonfinite_predictions():
    rows = [dict(prefix=prefix,entry_id='a',bce=.5,predicted_recall=float('nan'),recall=.5)
            for prefix in PREFIXES]
    with pytest.raises(ValueError,match='Nonfinite'):
        candidate_objectives(pd.DataFrame(rows))
