# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Argparse-surface smoke tests for the contacts-v1 CLI (no pyconfind run)."""

from pathlib import Path

import pytest

from marinfold.document_structures.contacts_v1 import cli


def test_generate_parses():
    args = cli.build_parser().parse_args([
        "generate", "--input", "cifs/", "--out", "docs.parquet",
        "--summary-out", "summary.json", "--num-docs", "10",
    ])
    assert args.cmd == "generate"
    assert args.func is cli.cmd_generate
    assert args.input == Path("cifs/")
    assert args.out == Path("docs.parquet")
    assert args.summary_out == Path("summary.json")
    assert args.num_docs == 10
    # pyconfind defaults wired through.
    assert args.native_only is True
    assert args.contact_distance == 3.0
    assert args.dcut == 25.0


def test_generate_requires_out():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--input", "cifs/"])


def test_no_native_only_flag():
    args = cli.build_parser().parse_args([
        "generate", "--input", "x", "--out", "o.jsonl", "--no-native-only",
    ])
    assert args.native_only is False


def test_view_parses_with_max_contacts():
    args = cli.build_parser().parse_args([
        "view", "--input", "tests/data/1QYS.cif", "--max-contacts", "5",
    ])
    assert args.cmd == "view"
    assert args.func is cli.cmd_view
    assert args.max_contacts == 5


def test_view_default_max_contacts():
    args = cli.build_parser().parse_args(["view", "--input", "x"])
    assert args.max_contacts == 20


def test_tokenizer_parses():
    args = cli.build_parser().parse_args([
        "tokenizer", "--save-local", "./tok", "--private",
    ])
    assert args.cmd == "tokenizer"
    assert args.func is cli.cmd_tokenizer
    assert args.save_local == Path("./tok")
    assert args.private is True


def test_subcommand_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_top_level_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in ("generate", "view", "tokenizer"):
        assert sub in out
