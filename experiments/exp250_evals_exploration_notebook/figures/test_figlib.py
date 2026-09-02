# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""What `figlib` promises the figure notebooks, checked.

The case worth a test is the one that has no error message: helico exp14 publishes seven arms
today, #250 re-ran one of them and keeps that arm's rows in this repo, and helico will eventually
republish with the re-run arm included. On that day a plain concatenation of the two sources gives
every one of the arm's targets twice — the means do not move, but n doubles and the bootstrap
intervals shrink. A figure drawn from that looks fine and claims more confidence than it has.

    uv run --with pytest pytest test_figlib.py
"""

import pandas as pd
import pytest

import figlib

COLUMNS = ["target_id", "arm", "status", "lddt", "gdt_ts", "designed"]


def table(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def published(monkeypatch):
    """Stand in for the published per-target table, whatever it happens to contain."""
    def install(frame):
        monkeypatch.setattr(figlib, "fetch",
                            lambda url: frame.to_csv(index=False).encode())
    return install


def local(tmp_path, frame):
    path = tmp_path / "per_target.csv"
    frame.to_csv(path, index=False)
    return path


def test_the_rerun_arm_is_added_while_it_is_unpublished(tmp_path, published):
    published(table([("7abc_A", "off", "ok", 0.35, 0.15, 0)]))
    extra = table([("7abc_A", "mf_L_363k", "ok", 0.64, 0.51, 0)])
    frame = figlib.load_helico_per_target(figlib.Inputs(), local(tmp_path, extra))
    assert sorted(frame.arm) == ["mf_L_363k", "off"]


def test_the_published_copy_wins_once_helico_republishes(tmp_path, published, capsys):
    """The transition: the same arm in both sources must not be counted twice."""
    both = table([("7abc_A", "off", "ok", 0.35, 0.15, 0),
                  ("7abc_A", "mf_L_363k", "ok", 0.64, 0.51, 0)])
    published(both)
    extra = table([("7abc_A", "mf_L_363k", "ok", 0.64, 0.51, 0)])
    frame = figlib.load_helico_per_target(figlib.Inputs(), local(tmp_path, extra))
    assert len(frame) == len(both)
    assert not frame.duplicated(["arm", "target_id"]).any()
    assert "redundant" in capsys.readouterr().out


def test_a_partial_republish_keeps_only_the_targets_that_are_missing(tmp_path, published):
    published(table([("7abc_A", "mf_L_363k", "ok", 0.64, 0.51, 0)]))
    extra = table([("7abc_A", "mf_L_363k", "ok", 0.64, 0.51, 0),
                   ("8def_B", "mf_L_363k", "ok", 0.70, 0.55, 0)])
    frame = figlib.load_helico_per_target(figlib.Inputs(), local(tmp_path, extra))
    assert sorted(frame.target_id) == ["7abc_A", "8def_B"]


def test_a_column_mismatch_is_refused_rather_than_filled(tmp_path, published):
    published(table([("7abc_A", "off", "ok", 0.35, 0.15, 0)]))
    extra = pd.DataFrame([("7abc_A", "mf_L_363k", "ok")], columns=["target_id", "arm", "status"])
    with pytest.raises(SystemExit, match="lddt"):
        figlib.load_helico_per_target(figlib.Inputs(), local(tmp_path, extra))
