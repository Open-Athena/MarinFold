# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from verify_linclust_pairs import build_numeric_candidate_tsv, db_ready, parameter_tag


def test_build_numeric_candidate_tsv_enumerates_intra_star_pairs(tmp_path: Path) -> None:
    source = tmp_path / "clusters.tsv"
    source.write_text("2\t2\n2\t0\n2\t1\n3\t3\n")
    output = tmp_path / "pairs.tsv"

    pairs, membership_rows, stars = build_numeric_candidate_tsv(source, output)

    assert pairs == 3
    assert membership_rows == 4
    assert stars == 2
    assert output.read_text() == "2\t0\t0\t0\n2\t1\t0\t0\n0\t1\t0\t0\n"


def test_parameter_tag_matches_candidate_generation() -> None:
    assert parameter_tag(0.5, 0.8, 1000.0) == "id050_cov080_e1000p0"


def test_db_ready_accepts_split_mmseqs_database(tmp_path: Path) -> None:
    prefix = tmp_path / "cluster_db"
    Path(f"{prefix}.0").touch()
    prefix.with_suffix(".dbtype").touch()

    assert db_ready(prefix)
