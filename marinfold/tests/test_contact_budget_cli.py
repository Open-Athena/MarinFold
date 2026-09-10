# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marinfold import cli
from marinfold.document_structures import contacts_v1, contacts_and_distances_v1
from marinfold.document_structures.contacts_v1.cli import (
    _inference_config, build_parser,
)


@pytest.mark.parametrize("command", ["infer", "evaluate"])
def test_top_level_contact_budget_config(command: str) -> None:
    args = cli.build_parser().parse_args([
        command, "--input", "example.cif", "--out", "out.json",
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


def test_other_document_structure_rejects_contact_budget() -> None:
    args = cli.build_parser().parse_args([
        "infer", "--input-sequence", "ACDEFGHIK", "--out", "out.json",
        "--min-new-contacts", "30",
    ])
    with pytest.raises(SystemExit, match="not supported"):
        cli._make_inference_config(contacts_and_distances_v1, None, args)
