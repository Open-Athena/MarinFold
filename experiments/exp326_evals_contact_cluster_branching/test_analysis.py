"""Unit checks for primary continuation-only contact scoring."""

import pandas as pd
from analyze_natural import branch_maps, canonical_contacts, rollout_r_precision

RECORD = {"resolved": list(range(40))}


def test_branch_primary_removes_supplied_contacts_but_complete_restores_them() -> None:
    frame = pd.DataFrame(
        [
            {
                "rollout": 0,
                "seed_contacts": [[1, 10], [2, 30]],
                "contacts": [[1, 10], [3, 20], [2, 30]],
            }
        ]
    )
    continuation, repeat = branch_maps(frame, RECORD, "all", include_seed=False)
    complete, _ = branch_maps(frame, RECORD, "all", include_seed=True)
    assert continuation == [[(3, 20)]]
    assert complete == [[(1, 10), (2, 30), (3, 20)]]
    assert repeat == 1.0


def test_fixed_r_charges_unfilled_predictions() -> None:
    truth = {(1, 10), (2, 20), (3, 30)}
    assert rollout_r_precision([(1, 10)], truth) == 1 / 3


def test_canonical_contacts_filters_unresolved_and_short_range() -> None:
    record = {"resolved": [0, 1, 10, 20]}
    assert canonical_contacts([[10, 1], [0, 1], [2, 20], [1, 10]], record, "all") == [
        (1, 10)
    ]
