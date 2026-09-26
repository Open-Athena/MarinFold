# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from build_current_fastas import fasta_records, parse_header


def test_parse_header_preserves_underscores_in_entry_id():
    assert parse_header(">esm_atlas|00312_9_entry_with_underscores\n") == (
        "esm_atlas",
        "entry_with_underscores",
    )


@pytest.mark.parametrize("header", [">bad", ">unknown|00001_2_x", ">afdb|missing"])
def test_parse_header_rejects_invalid_grammar(header):
    with pytest.raises(ValueError, match="invalid #213 FASTA header"):
        parse_header(header)


def test_fasta_records_preserves_wrapped_lines(tmp_path: Path):
    path = tmp_path / "tiny.fasta"
    path.write_text(">afdb|00000_0_a\nABC\nDEF\n>afdb|00000_1_b\nGHI\n")
    assert list(fasta_records(path)) == [
        (">afdb|00000_0_a\n", ["ABC\n", "DEF\n"]),
        (">afdb|00000_1_b\n", ["GHI\n"]),
    ]
