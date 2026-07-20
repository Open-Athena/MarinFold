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


def test_think_defaults_off():
    args = cli.build_parser().parse_args(["generate", "--input", "x", "--out", "o.jsonl"])
    cfg = cli._config_from_args(args)
    assert cfg.think is False
    # #123 distribution defaults are wired through even while off.
    assert cfg.think_initial_prob == 0.75
    assert cfg.think_initial_geom_p == 0.13
    assert cfg.think_additional_count_range == (-4.0, 4.0)
    assert cfg.think_run_length_geom_p == 0.25


def test_think_flag_and_overrides():
    args = cli.build_parser().parse_args([
        "generate", "--input", "x", "--out", "o.jsonl", "--think",
        "--think-initial-prob", "1.0",
        "--think-additional-count-range", "-2", "6",
        "--think-run-length-geom-p", "0.5",
    ])
    cfg = cli._config_from_args(args)
    assert cfg.think is True
    assert cfg.think_initial_prob == 1.0
    assert cfg.think_additional_count_range == (-2.0, 6.0)
    assert cfg.think_run_length_geom_p == 0.5


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
    for sub in ("generate", "view", "infer", "evaluate", "tokenizer"):
        assert sub in out


def test_infer_parses_input_sequence():
    args = cli.build_parser().parse_args([
        "infer", "--model", "contacts-v1-exp75-1.5B",
        "--input-sequence", "SIINFEKLLLSKP", "--out", "preds.json",
        "--ensemble-k", "10",
    ])
    assert args.cmd == "infer"
    assert args.func is cli.cmd_infer
    assert args.input_sequence == "SIINFEKLLLSKP"
    assert args.input is None
    assert args.ensemble_k == 10
    assert args.min_seq_separation == 6  # default
    cfg = cli._inference_config(args)
    assert cfg.ensemble_k == 10
    assert cfg.model == "contacts-v1-exp75-1.5B"


def test_infer_requires_a_source():
    """--input-sequence / --input are mutually exclusive and one is required."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "infer", "--model", "M", "--out", "preds.json",
        ])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "infer", "--model", "M", "--out", "preds.json",
            "--input-sequence", "ACDE", "--input", "x.cif",
        ])


def test_infer_keep_matrix_flag():
    args = cli.build_parser().parse_args([
        "infer", "--model", "M", "--input-sequence", "ACDEFGHIK",
        "--out", "preds.parquet", "--keep-matrix",
    ])
    assert args.keep_matrix is True


def test_infer_method_defaults_pairwise():
    args = cli.build_parser().parse_args([
        "infer", "--model", "M", "--input-sequence", "ACDE", "--out", "p.json",
    ])
    assert args.method == "pairwise"
    assert cli._inference_config(args).method == "pairwise"


def test_infer_method_rollout_flags():
    args = cli.build_parser().parse_args([
        "infer", "--model", "M", "--input-sequence", "ACDEFGHIK",
        "--out", "preds.json", "--method", "rollout",
        "--n-rollouts", "50", "--temperature", "0.7", "--top-p", "0.9", "--top-k", "0",
    ])
    assert args.method == "rollout"
    cfg = cli._inference_config(args)
    assert cfg.method == "rollout"
    assert cfg.n_rollouts == 50
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.9
    assert cfg.top_k == 0


def test_infer_method_rejects_unknown():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "infer", "--model", "M", "--input-sequence", "ACDE",
            "--out", "p.json", "--method", "bogus",
        ])


def test_evaluate_parses_and_rejects_input_sequence():
    args = cli.build_parser().parse_args([
        "evaluate", "--model", "M", "--input", "tests/data/1QYS.cif",
        "--out", "metrics.json", "--out-plots", "heatmaps.pdf",
    ])
    assert args.cmd == "evaluate"
    assert args.func is cli.cmd_evaluate
    assert args.out_plots == Path("heatmaps.pdf")
    # evaluate has no --input-sequence (ground truth needs a structure).
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "evaluate", "--model", "M", "--input-sequence", "ACDE",
            "--out", "metrics.json",
        ])


def test_out_plots_must_be_pdf():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "infer", "--model", "M", "--input-sequence", "ACDE",
            "--out", "preds.json", "--out-plots", "heatmaps.png",
        ])
