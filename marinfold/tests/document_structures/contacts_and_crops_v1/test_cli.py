# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Argparse-surface tests for the CLI (no pyconfind run)."""

from pathlib import Path

import pytest

from marinfold.document_structures.contacts_and_crops_v1 import cli
from marinfold.document_structures.contacts_and_crops_v1.vocab import CONTEXT_LENGTH


def test_generate_defaults_wire_through():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--input", "x.cif", "--out", "y.parquet"])
    assert args.input == Path("x.cif")
    assert args.out == Path("y.parquet")
    assert args.context_length == CONTEXT_LENGTH == 8192
    config = cli._config_from_args(args)
    assert config.cube_size == 1000.0
    assert config.fine_reserve == 2000
    assert config.pass1_box_noise_sigma == 2.0
    assert config.pass2_select_random == 0.45
    assert config.pass2_select_frontier == 0.45
    assert config.pass2_keep_prob == 0.99
    assert config.n_contacts_zero_prob == 0.3
    assert config.n_contacts_max == 50


def test_crop_knobs_override():
    parser = cli.build_parser()
    args = parser.parse_args([
        "view", "--input", "x.cif",
        "--fine-reserve", "1500", "--pass2-select-random", "0.3",
        "--pass2-select-frontier", "0.3", "--pass1-box-noise-sigma", "1.0",
        "--context-length", "4096",
    ])
    config = cli._config_from_args(args)
    assert config.fine_reserve == 1500
    assert config.pass2_select_random == 0.3
    assert config.pass2_select_frontier == 0.3
    assert config.pass1_box_noise_sigma == 1.0
    assert args.context_length == 4096
    # Derived re-show probability follows the two overrides.
    assert abs(config.pass2_select_reshow - 0.4) < 1e-9


@pytest.mark.parametrize(("raw", "expected"), [
    ("none", None), ("None", None), ("1", 1), ("assembly-a", "assembly-a"),
])
def test_assembly_arg(raw, expected):
    assert cli._assembly_arg(raw) == expected


def test_tokenizer_subcommand_builds(capsys):
    cli.main(["tokenizer"])
    out = capsys.readouterr()
    assert "built tokenizer with 3848 tokens" in out.err


def test_missing_required_args_exit():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--input", "x.cif"])  # missing --out
