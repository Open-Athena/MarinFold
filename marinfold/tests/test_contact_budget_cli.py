# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marinfold import cli
from marinfold.document_structures import contacts_v1
from marinfold.document_structures.contacts_v1.cli import (
    _inference_config, build_parser,
)


@pytest.mark.parametrize("command", ["infer", "evaluate"])
def test_top_level_contact_budget_config(command: str, tmp_path) -> None:
    args = cli.parse_args([
        command, "--input", "example.cif", "--out", "out.json",
        "--model", str(tmp_path), "--document-structure", "contacts-v1",
        "--method", "rollout", "--n-rollouts", "2", "--min-new-contacts", "30",
    ])
    cfg = cli._make_inference_config(contacts_v1, None, args)
    assert cfg.method == "rollout"
    assert cfg.n_rollouts == 2
    assert cfg.min_new_contacts == 30


def test_per_structure_contact_budget_config(tmp_path) -> None:
    args = build_parser().parse_args([
        "infer", "--model", str(tmp_path), "--input-sequence", "ACDEFGHIK",
        "--out", "out.json", "--method", "rollout", "--min-new-contacts", "30",
    ])
    assert _inference_config(args).min_new_contacts == 30


def test_default_config_unchanged() -> None:
    args = cli.build_parser().parse_args([
        "infer", "--input-sequence", "ACDEFGHIK", "--out", "out.json",
    ])
    cfg = cli._make_inference_config(contacts_v1, None, args)
    assert cfg.method == "pairwise"
    assert cfg.min_new_contacts is None


def test_pairwise_rejects_contact_budget() -> None:
    with pytest.raises(ValueError, match="method='rollout'"):
        contacts_v1.InferenceConfig(model=None, min_new_contacts=30)


def test_other_document_structure_rejects_contact_budget(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.parse_args([
            "infer", "--input-sequence", "ACDEFGHIK", "--out", "out.json",
            "--model", str(tmp_path), "--document-structure", "contacts-and-distances-v1",
            "--min-new-contacts", "30",
        ])
    assert error.value.code == 2
    assert "unrecognized arguments: --min-new-contacts" in capsys.readouterr().err


@pytest.mark.parametrize("structure, exposes_budget", [
    ("contacts-v1", True), ("contacts-and-distances-v1", False),
])
def test_selected_format_controls_help(structure, exposes_budget, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([
            "infer", "--model", str(tmp_path), "--document-structure", structure, "--help",
        ])
    assert error.value.code == 0
    assert ("--min-new-contacts" in capsys.readouterr().out) == exposes_budget


@pytest.mark.parametrize("model_args", [[], ["--model", "contacts-v1-exp75-1.5B"]])
def test_nickname_selects_format_arguments(model_args: list[str]) -> None:
    args = cli.parse_args([
        "infer", *model_args, "--input-sequence", "ACDEFGHIK", "--out", "out.json",
        "--method", "rollout", "--min-new-contacts", "30",
    ])
    assert args.min_new_contacts == 30
