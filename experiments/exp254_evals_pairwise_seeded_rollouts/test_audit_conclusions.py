# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from audit_conclusions import bootstrap, rollout_scores


def test_short_perfect_rollout_does_not_fill_fixed_r_budget():
    truth = np.zeros((12, 12), dtype=bool)
    truth[0, 6] = truth[1, 7] = True
    contacts = pd.DataFrame(dict(rollout=[0], rank=[0], i=[0], j=[6], is_seed=[False]))
    result = rollout_scores(contacts, truth, 2, np.array([0, 1]), False)
    assert result.legacy_precision.tolist() == [1.0, 0.0]
    assert result.fixed_R.tolist() == [0.5, 0.0]
    assert result.n_emitted.tolist() == [1, 0]


def test_seed_removed_before_rank_cut_and_empty_continuation_retained():
    truth = np.zeros((12, 12), dtype=bool)
    truth[0, 6] = truth[2, 8] = True
    # Rollout 0 has a wrong seed before two correct predictions. Rollout 1 only
    # contains a true seed, which must not become a perfect empty continuation.
    contacts = pd.DataFrame(dict(rollout=[0, 0, 0, 1], rank=[0, 2, 1, 0],
                                 i=[1, 2, 0, 0], j=[7, 8, 6, 6],
                                 is_seed=[True, False, False, True]))
    result = rollout_scores(contacts, truth, 2, np.array([0, 1]), True)
    assert result.fixed_R.tolist() == [1.0, 0.0]
    assert result.n_emitted.tolist() == [2, 0]
    assert result.legacy_precision.tolist() == [1.0, 0.0]


def test_bootstrap_does_not_silently_drop_missing_proteins():
    with pytest.raises(ValueError, match="finite"):
        bootstrap(np.array([0.1, float("nan")]))
