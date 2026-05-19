# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

import pytest

from marinfold.registry import (
    _locate_models_yaml,
    default_model_nickname,
    list_model_entries,
    resolve_model_entry,
)


def test_locate_models_yaml_falls_back_to_package_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nickname resolution should not depend on the caller's cwd being in-repo."""
    monkeypatch.delenv("MARINFOLD_MODELS_YAML", raising=False)
    monkeypatch.chdir(Path("/tmp"))

    yaml_path = _locate_models_yaml()

    assert yaml_path.name == "MODELS.yaml"
    assert yaml_path.parent == Path(__file__).resolve().parents[2]


def test_repo_models_yaml_has_exactly_one_default() -> None:
    """Validates the checked-in MODELS.yaml; also exercises list_model_entries."""
    entries = list_model_entries()
    defaults = [e for e in entries if e.default]
    assert len(defaults) == 1, f"expected exactly one default; got {defaults}"
    assert default_model_nickname() == defaults[0].nickname


def test_resolve_model_entry_by_nickname_and_default() -> None:
    by_default = resolve_model_entry(None)
    by_name = resolve_model_entry(by_default.nickname)
    assert by_default == by_name


def test_resolve_model_entry_unknown_nickname_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="not found in MODELS.yaml"):
        resolve_model_entry("definitely-not-a-real-nickname-xyz")


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def test_multiple_defaults_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "MODELS.yaml"
    _write_yaml(yaml_path, """
        - nickname: A
          default: true
          url: https://huggingface.co/x/y/tree/main/sub
        - nickname: B
          default: true
          url: https://huggingface.co/x/z/tree/main/sub
    """)
    monkeypatch.setenv("MARINFOLD_MODELS_YAML", str(yaml_path))
    with pytest.raises(ValueError, match="multiple entries marked default"):
        list_model_entries()


def test_no_default_means_default_lookup_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "MODELS.yaml"
    _write_yaml(yaml_path, """
        - nickname: A
          url: https://huggingface.co/x/y/tree/main/sub
    """)
    monkeypatch.setenv("MARINFOLD_MODELS_YAML", str(yaml_path))
    with pytest.raises(LookupError, match="default: true"):
        default_model_nickname()
    with pytest.raises(LookupError, match="default: true"):
        resolve_model_entry(None)
