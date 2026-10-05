# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from summarize_direct_pairs import summarize


def test_summarize_validates_direct_central_link(tmp_path: Path) -> None:
    alignments = tmp_path / "direct.tsv"
    alignments.write_text(
        "afdb|a\tesm_atlas|b\t0.5\t100\t1\t100\t100\t1\t100\t100\t1e-5\t50\t0.8\t0.8\n"
    )
    memberships = tmp_path / "memberships.tsv"
    memberships.write_text("afdb|a\tafdb|a\nafdb|a\tesm_atlas|b\n")

    relationships, central_relationships, summary = summarize(
        alignments, memberships, 0.5, 0.8
    )

    assert relationships.to_dict("records") == [
        {"source_a": "afdb", "source_b": "esm_atlas", "directly_verified_pairs": 1}
    ]
    assert central_relationships.to_dict("records") == [
        {
            "representative_source": "afdb",
            "member_source": "esm_atlas",
            "linclust_candidates": 1,
            "directly_verified_candidates": 1,
        }
    ]
    assert summary["linclust_central_candidates"] == 1
    assert summary["directly_verified_central_candidates"] == 1
    assert summary["failed_direct_central_candidates"] == 0
