# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""What the Top7 animation claims, checked against its own dataset.

Two things in `7_plot_rollout_animation.py` are wrong in ways that look right.

The **mirroring**: with `origin="lower"` the visually upper-left triangle is `i > j`, so the
rollout is written to `canvas[j, i]` and the experimental contacts to the transpose. Swap them and
the figure is mirrored while every label still reads correctly — which is invisible until someone
compares it against figure 1.

The **running-precision curve**: it recovers the pairwise tie-break as `score - votes` and re-adds
it to a partial vote matrix, on the argument that the tie-break does not depend on the rollouts.
If either half of that were wrong the curve would still be a smooth, plausible line — it would
just not be the metric the repository quotes. So the recovered term has to be the bounded
fraction #82's recipe defines, and the curve's last point has to land on the R-precision the make
step wrote into `metrics.csv`. The curve's *first* point is where this matters most: after one
rollout nearly every emitted pair has exactly one vote, and the tie-break is what orders them.

    uv run --with pytest pytest test_rollout_animation.py
"""

import importlib.util
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

import figlib

STATEMENTS_CHECKED = 7   # frames are rasterised one per statement; a handful is the whole claim

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    """Import a script whose filename starts with a digit, which `import` cannot spell."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


animation = load_module("rollout_animation", "7_plot_rollout_animation.py")


@pytest.fixture(scope="module")
def data():
    """The committed dataset. Skipped rather than failed where it has not been generated."""
    if not (figlib.dataset_dir(animation.DATASET) / "metadata.json").exists():
        pytest.skip(f"{animation.DATASET} has not been generated — "
                    "run 7_make_rollout_animation_data.py (needs a GPU)")
    return animation.load_dataset()


def test_the_rollout_and_the_structure_are_on_opposite_sides_of_the_diagonal(data):
    canvas = animation.base_canvas(data)
    painted = np.any(canvas != 1.0, axis=-1)
    index = np.arange(data["length"])
    band = np.abs(np.subtract.outer(index, index)) < figlib.MIN_SEPARATION
    # Outside the unscorable band, `base_canvas` draws the experimental contacts and nothing
    # else, and every one of them is at i < j: the visually lower-right triangle.
    assert not painted[~band & np.tril(np.ones_like(painted), k=-1).astype(bool)].any()

    frame = animation.Frame(data, "test", right="stream")
    rollout = int(data["statements"].rollout.iloc[0])
    # The opening frame carries no statement, so n + 1 frames cover n statements.
    frames = animation.emission_frames(frame, data, rollout, slow_statements=0)
    for _ in itertools.islice(frames, STATEMENTS_CHECKED + 1):
        pass
    drawn = np.asarray(frame.image.get_array())
    plt.close(frame.figure)

    emitted = (data["statements"][data["statements"].rollout == rollout]
               .sort_values("order").head(STATEMENTS_CHECKED))
    assert emitted.scorable.all(), "the fixture's first statements are not all scorable"
    for row in emitted.itertuples(index=False):
        colour = to_rgb(animation.HIT if row.hit else animation.MISS)
        assert np.allclose(drawn[row.seq_j, row.seq_i], colour), (
            f"({row.seq_i}, {row.seq_j}) is not in the rollout's triangle")


def test_the_recovered_tiebreak_only_orders_pairs_that_are_tied(data):
    """`score - votes` has to land in [0, 0.5] over the candidate pairs, or it crosses vote gaps.

    Vote counts are integers, so a term bounded by half a vote can only reorder pairs already
    tied — that bound is the whole reason the vote matrix and the ranking can be stored as one
    number and pulled back apart here. The diagonal is excluded: it is never a candidate pair, and
    #82's min-max runs over the upper triangle, which does not contain it.
    """
    upper = np.triu_indices(data["length"], k=1)
    recovered = data["tiebreak"][upper]
    assert recovered.min() >= 0.0
    assert recovered.max() <= 0.5


def test_the_curve_ends_on_the_r_precision_the_make_step_recorded(data):
    order = list(range(data["n_rollouts"]))
    curve = animation.running_precision(data, order)
    metrics = pd.read_csv(figlib.dataset_dir(animation.DATASET) / "metrics.csv")
    published = float(metrics[(metrics.range == "all") & (metrics.cut == "R")].value.iloc[0])
    assert curve[-1] == pytest.approx(published, abs=1e-9)
    assert len(curve) == data["n_rollouts"]


def test_every_rollout_votes_once_per_distinct_pair(data):
    """The vote matrix the make step wrote is what replaying the statements produces."""
    counted = data["statements"][data["statements"].scorable & ~data["statements"].duplicate]
    votes = np.zeros_like(data["votes"])
    for i, j in counted[["seq_i", "seq_j"]].itertuples(index=False):
        votes[i, j] += 1
        votes[j, i] += 1
    assert np.array_equal(votes, data["votes"])
    assert votes.max() <= data["n_rollouts"]
