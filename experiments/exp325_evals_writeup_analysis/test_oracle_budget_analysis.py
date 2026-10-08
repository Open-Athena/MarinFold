"""Confidence selection must stay independent of structural accuracy."""

import pandas as pd

from prepare_oracle_budget_analysis import select_samples


def test_selector_uses_confidence_and_keeps_every_random_subset() -> None:
    rows = []
    for replicate in (0, 1):
        for sample, rank, ptm, accuracy in [(0, 0.5, 0.8, 0.95), (1, 0.9, 0.7, 0.2), (2, 0.6, 0.8, 0.8)]:
            rows.append(dict(stem="protein", arm="random_10", map_seed=replicate,
                sample_idx=sample, ranking_score=rank, ptm=ptm, gdt_ts=accuracy))
    samples = pd.DataFrame(rows)
    ranking = select_samples(samples, "ranking_score")
    ptm = select_samples(samples, "ptm")
    assert len(ranking) == len(ptm) == 2
    assert set(ranking.sample_idx) == {1}
    assert set(ptm.sample_idx) == {0}
    changed = samples.assign(gdt_ts=1 - samples.gdt_ts)
    assert list(select_samples(changed, "ranking_score").sample_idx) == list(ranking.sample_idx)
