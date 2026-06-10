# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Argparse-surface smoke tests for the contacts-v1 CLI (no pyconfind run)."""

from pathlib import Path

import pytest

from marinfold.document_structures.contacts_v1 import cli
from marinfold.document_structures.contacts_v1.generate import GenerationConfig, build_document
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_v1.vocab import position_token


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
    assert args.assembly is None
    # pyconfind defaults wired through.
    assert args.native_only is True
    assert args.contact_distance == 3.0
    assert args.dcut == 25.0
    assert args.min_contact_degree == 0.001


def test_min_contact_degree_override():
    args = cli.build_parser().parse_args([
        "generate", "--input", "x", "--out", "o.jsonl",
        "--min-contact-degree", "0.05",
    ])
    assert args.min_contact_degree == 0.05
    cfg = cli._config_from_args(args)
    assert cfg.min_contact_degree == 0.05


def test_min_seq_separation_default_and_override():
    args = cli.build_parser().parse_args(["generate", "--input", "x", "--out", "o.jsonl"])
    assert args.min_seq_separation == 6
    assert cli._config_from_args(args).min_seq_separation == 6

    args2 = cli.build_parser().parse_args([
        "generate", "--input", "x", "--out", "o.jsonl", "--min-seq-separation", "12",
    ])
    assert args2.min_seq_separation == 12
    assert cli._config_from_args(args2).min_seq_separation == 12


def test_parquet_column_defaults():
    args = cli.build_parser().parse_args([
        "generate", "--input", "shard.parquet", "--out", "docs.parquet",
    ])
    assert args.cif_column == "cif_content"
    assert args.id_column == "entry_id"


def test_parquet_column_overrides():
    args = cli.build_parser().parse_args([
        "generate", "--input", "shard.parquet", "--out", "docs.parquet",
        "--cif-column", "mmcif", "--id-column", "id",
    ])
    assert args.cif_column == "mmcif"
    assert args.id_column == "id"


@pytest.mark.parametrize(("raw", "expected"), [
    ("none", None),
    ("2", 2),
    ("bio1", "bio1"),
])
def test_generate_parses_assembly(raw, expected):
    args = cli.build_parser().parse_args([
        "generate", "--input", "cifs/", "--out", "docs.parquet",
        "--assembly", raw,
    ])
    assert args.assembly == expected


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


def test_view_prints_reused_position_tokens(monkeypatch, capsys):
    residues = [
        ResidueInfo(seq_index=i, resname=aa, resnum=1 + i, chain="A")
        for i, aa in enumerate(
            ["ALA", "GLY", "SER", "THR", "LYS", "VAL", "LEU", "PHE"]
        )
    ]
    # sep 6 → survives the default min_seq_separation=6 filter.
    result = build_document("view-demo", residues, [RawContact(0, 6, 0.9)])
    assert result is not None
    assert result.contacts_emitted == 1
    monkeypatch.setattr(
        cli.generate,
        "generate_documents",
        lambda **_: iter([result]),
    )

    args = cli.build_parser().parse_args(["view", "--input", "demo.cif", "--max-contacts", "5"])
    args.func(args)

    out = capsys.readouterr().out
    assert f"n_term={position_token(result.n_term_index)}" in out
    assert f"c_term={position_token(result.c_term_index)}" in out
    assert (
        f"{position_token(result.contacts[0].pos_i)}/"
        f"{position_token(result.contacts[0].pos_j)}"
    ) in out
    assert "<pos-" not in out


def test_view_handles_no_included_contacts(monkeypatch, capsys):
    residues = [
        ResidueInfo(seq_index=i, resname=aa, resnum=1 + i, chain="A")
        for i, aa in enumerate(["ALA", "GLY", "SER", "THR"])
    ]
    result = build_document(
        "view-empty-contacts",
        residues,
        [RawContact(0, 2, 0.9)],
        config=GenerationConfig(min_seq_separation=1, min_contact_degree=1.0),
    )
    assert result is not None
    assert result.contacts_pre_filter == 1
    assert result.contacts_emitted == 0
    monkeypatch.setattr(
        cli.generate,
        "generate_documents",
        lambda **_: iter([result]),
    )

    args = cli.build_parser().parse_args(["view", "--input", "demo.cif"])
    args.func(args)

    out = capsys.readouterr().out
    assert "survive seq-sep>=1" in out
    assert "lowest_included=n/a" in out


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
