# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from marinfold_inference.registry import _locate_models_yaml


def test_locate_models_yaml_falls_back_to_package_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nickname resolution should not depend on the caller's cwd being in-repo."""
    monkeypatch.delenv("MARINFOLD_MODELS_YAML", raising=False)
    monkeypatch.chdir(Path("/tmp"))

    yaml_path = _locate_models_yaml()

    assert yaml_path.name == "MODELS.yaml"
    assert yaml_path.parent == Path(__file__).resolve().parents[2]
