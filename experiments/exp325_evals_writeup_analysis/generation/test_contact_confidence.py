"""Checks against confidence leakage, invented contacts, and range-order bugs."""

import numpy as np
import pytest

from score_contact_confidence import rank_contacts


def test_confidence_rankings_change_order_but_never_invent_contacts() -> None:
    i = np.array([0, 0, 0, 0])
    j = np.array([6, 7, 8, 9])
    degree = np.array([0.9, 0.3, 0.4, 0.0])
    confidence = np.ones(10)
    confidence[j] = [0.1, 0.9, 0.8, 1.0]
    expected = {"degree":[0,2,1], "plddt_product":[1,2,0], "plddt_product_degree":[2,1,0]}
    for ranking, order in expected.items():
        actual = rank_contacts(i,j,degree,confidence,minimum=6,k=5,ranking=ranking)
        assert actual.tolist() == order


def test_filter_long_range_before_selecting_budget() -> None:
    i = np.array([0, 0, 1])
    j = np.array([7, 24, 30])
    degree = np.array([0.99, 0.2, 0.3])
    confidence = np.ones(31)
    for ranking in ["degree","plddt_product","plddt_product_degree"]:
        assert set(rank_contacts(i,j,degree,confidence,minimum=24,k=2,ranking=ranking)) == {1,2}


def test_confidence_ties_use_pair_order_and_keep_zero_confidence_contacts() -> None:
    i, j = np.array([0,0,1]), np.array([8,9,9])
    degree = np.array([0.01,0.99,0.5])
    confidence = np.zeros(10)
    result = rank_contacts(i,j,degree,confidence,minimum=6,k=2,ranking="plddt_product")
    assert result.tolist() == [0,1]


def test_missing_contact_confidence_fails() -> None:
    confidence = np.ones(10)
    confidence[8] = np.nan
    with pytest.raises(ValueError, match="lacks mapped confidence"):
        rank_contacts(np.array([0]),np.array([8]),np.array([0.2]),confidence,minimum=6,k=1,ranking="plddt_product")
