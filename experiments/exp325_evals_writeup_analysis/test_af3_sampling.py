"""Selection must use only available-prefix confidence, never future truth."""

import pandas as pd
import pytest

from prepare_af3_sampling import prefix_curves


def test_confidence_selection_can_get_less_accurate_and_breaks_ties_by_seed() -> None:
    frame = pd.DataFrame([
        dict(stem="p", seed=102, msa_depth=3, tm_score=0.95, ptm=0.8, ranking_score=0.7),
        dict(stem="p", seed=100, msa_depth=3, tm_score=0.85, ptm=0.6, ranking_score=0.9),
        dict(stem="p", seed=101, msa_depth=3, tm_score=0.3, ptm=0.8, ranking_score=0.8),
    ])
    curves = prefix_curves(frame, 100, 0.8)
    assert curves.best_tm.tolist() == [0.85, 0.85, 0.95]
    assert curves.ptm_selected_tm.tolist() == [0.85, 0.3, 0.3]
    assert curves.ranking_selected_seed.tolist() == [100, 100, 100]
    assert curves.hits_tm_80.tolist() == [1, 1, 2]
    assert curves.first_hit_draw.tolist() == [1, 1, 1]


@pytest.mark.parametrize("seeds", [[100, 102], [100, 100]])
def test_incomplete_or_duplicate_seed_prefix_fails(seeds: list[int]) -> None:
    frame = pd.DataFrame([dict(stem="p", seed=s, msa_depth=3, tm_score=0.4, ptm=0.5,
                               ranking_score=0.6) for s in seeds])
    with pytest.raises(ValueError, match="complete consecutive prefix"):
        prefix_curves(frame, 100, 0.8)
