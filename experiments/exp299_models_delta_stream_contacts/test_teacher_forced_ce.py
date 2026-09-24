import numpy as np

from score_teacher_forced_ce import (
    batches_by_token_budget,
    classify_targets,
    masked_mean,
)


def test_classify_delta_v2_targets() -> None:
    ids = [21, 1, 20, 22, 32, 0, 99, 23]
    assert classify_targets(ids, "delta-v2").tolist() == [1, 1, 0, 3, 4, 3, 0]


def test_classify_contacts_v1_targets() -> None:
    ids = [2, 8, 143, 86, 144, 2844, 3, 9, 143, 144, 5, 10, 1]
    assert classify_targets(ids, "contacts-v1").tolist() == [0, 2, 1, 2, 1, 2, 0, 3, 3, 3, 0, 0]


def test_batches_respect_padded_token_budget() -> None:
    documents = [
        (0, "long", [0] * 7, None, None),
        (1, "short", [0] * 3, None, None),
        (2, "medium", [0] * 5, None, None),
    ]
    batches = list(batches_by_token_budget(documents, token_budget=10))
    assert [[document[1] for document in batch] for batch in batches] == [["short", "medium"], ["long"]]
    assert all(max(len(document[2]) for document in batch) * len(batch) <= 10 for batch in batches)


def test_masked_mean() -> None:
    count, mean = masked_mean(np.asarray([1.0, 2.0, 7.0]), np.asarray([1, 3, 0]), {1, 3})
    assert count == 2
    assert mean == 1.5
