# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from select_sequence_witnesses import pair_key, select_all, select_star


def test_select_star_requires_a_retained_direct_witness() -> None:
    identities = {
        pair_key(1, 2): 0.8,
        pair_key(2, 3): 0.8,
    }

    kept, removals = select_star(1, [1, 2, 3], identities, 0.5)

    assert kept == [1, 3]
    assert removals == [(2, 1)]


def test_select_all_sweeps_thresholds_and_sources(tmp_path: Path) -> None:
    memberships = tmp_path / "memberships.tsv"
    memberships.write_text(
        "0\t0\n0\t1\n0\t65553178\n65553179\t65553179\n"
    )
    identities = {
        pair_key(0, 1): 0.8,
        pair_key(0, 65_553_178): 0.6,
        pair_key(1, 65_553_178): 0.55,
    }

    summary, relationships, rows, stars = select_all(
        memberships, identities, [0.5, 0.7]
    )

    assert rows == 4
    assert stars == 2
    assert summary["documents_removed"].tolist() == [2, 1]
    assert relationships["documents_removed"].sum() == 3
