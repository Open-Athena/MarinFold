# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Argparse-surface tests for the CLI (no pyconfind run)."""

from pathlib import Path

import pytest

from marinfold.document_structures.contacts_and_coordinates_v1 import cli
from marinfold.document_structures.contacts_and_coordinates_v1.vocab import CONTEXT_LENGTH


def test_generate_defaults_wire_through():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--input", "x.cif", "--out", "y.parquet"])
    assert args.input == Path("x.cif")
    assert args.out == Path("y.parquet")
    assert args.context_length == CONTEXT_LENGTH
    config = cli._config_from_args(args)
    # Coordinate-section defaults land in the config.
    assert config.cube_size == 1000.0
    assert config.cube_margin == 10.0
    assert config.max_depth == 3
    assert config.noise_divisor == 4.0
    assert config.n_contacts_zero_prob == 0.3
    assert config.n_contacts_max == 50


def test_coordinate_knobs_override():
    parser = cli.build_parser()
    args = parser.parse_args([
        "view", "--input", "x.cif",
        "--cube-size", "500", "--max-depth", "4",
        "--n-contacts-max", "10", "--context-length", "8192",
    ])
    config = cli._config_from_args(args)
    assert config.cube_size == 500.0
    assert config.max_depth == 4
    assert config.n_contacts_max == 10
    assert args.context_length == 8192


@pytest.mark.parametrize(("raw", "expected"), [
    ("none", None), ("None", None), ("1", 1), ("assembly-a", "assembly-a"),
])
def test_assembly_arg(raw, expected):
    assert cli._assembly_arg(raw) == expected


def test_tokenizer_subcommand_builds(capsys):
    # No --save-local / --push: just builds and prints a sample (no network).
    cli.main(["tokenizer"])
    out = capsys.readouterr()
    assert "built tokenizer with 3847 tokens" in out.err


def test_missing_required_args_exit():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--input", "x.cif"])  # missing --out
