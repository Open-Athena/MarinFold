"""Guard alignment projection against input leakage and correspondence errors."""

import pytest

from prepare_baselines import project_a3m


def test_msa_projection_preserves_only_relevant_insertions() -> None:
    msa = ">query\nACDEFG\n>homolog\nAaaCbbDccEddF-G\n"
    # The last row must contain exactly six match columns; extra gaps are real
    # alignment columns, not insertions, and must fail before prediction.
    with pytest.raises(ValueError, match="match-column"):
        project_a3m(msa, "ACDEFG", [1, 2, 4])
    msa = ">query\nACDEFG\n>homolog\nAaaCbbDccEddF-G\n".replace("F-G", "F-")
    assert project_a3m(msa, "ACDEFG", [1, 2, 4]) == ">query\nCDF\n>homolog\nCbbDF\n"


def test_wrong_query_is_rejected_before_projection() -> None:
    with pytest.raises(ValueError, match="query differs"):
        project_a3m(">query\nACDE\n", "AAAA", [0, 1])
