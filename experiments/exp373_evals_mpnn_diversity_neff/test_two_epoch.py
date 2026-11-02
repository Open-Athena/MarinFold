"""Exercise shared reduction and four-model pairing with known counterexamples."""

import numpy as np
import pandas as pd
import pytest

from extend_analysis import reduce_record
from two_epoch_summary import MODELS, paired_values


def test_oracle_can_improve_while_ensemble_stays_wrong() -> None:
    truth = {
        "stem": "example",
        "L": 30,
        "resolved": [0, 1, 24, 25],
        "contacts": [[0, 24, 0.5]],
    }
    samples = []
    for k in range(1000):
        i, j = (0, 24) if k == 999 else (1, 25)
        samples.append(
            dict(
                rollout=k, offset=0, text=f"<contact> <p{i}> <p{j}>", contacts=[[i, j]]
            )
        )
    record = dict(
        model="new", target={"stem": "example", "eval_set": "eval-val"}, samples=samples
    )
    curves, singles = reduce_record(record, truth, [300, 1000])
    for region in ["all", "long"]:
        before, after = [r for r in curves if r["range"] == region]
        assert before["oracle_f1"] == 0 and after["oracle_f1"] == 1
        assert before["r_precision"] == after["r_precision"] == 0
        assert after["mean_f1"] == 0.001 and after["false_discovered"] == 1
    assert len(singles) == 2000


def test_four_model_contrasts_pair_proteins_and_reject_missingness() -> None:
    frame = pd.DataFrame(
        [
            dict(stem=stem, model=model, score=value + index)
            for stem, value in [("b", 10), ("a", 1)]
            for index, model in enumerate(MODELS)
        ]
    )
    values = paired_values(frame.sample(frac=1, random_state=1), "score")
    np.testing.assert_array_equal(values["mpnn2-lr5e-4 minus default"], [2, 2])
    with pytest.raises(AssertionError):
        paired_values(frame.iloc[1:], "score")
