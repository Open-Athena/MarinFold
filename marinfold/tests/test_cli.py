# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import marinfold.cli as cli


@dataclass(frozen=True)
class _FakeInferenceConfig:
    model: str | None
    input_path: Path | None = None
    backend: str = "vllm"
    batch_size: int = 64
    dtype: str = "bfloat16"


def test_cmd_infer_accepts_local_model_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    out_path = tmp_path / "preds.json"
    captured: dict[str, Any] = {}

    def _predict(cfg: _FakeInferenceConfig, *, structures=None):
        captured["cfg"] = cfg
        captured["structures"] = structures
        yield {"entry_id": "sequence", "pairs": [], "expected_distances": []}

    fake_impl = SimpleNamespace(
        InferenceConfig=_FakeInferenceConfig,
        predict=_predict,
        structure_from_sequence=lambda seq: {"sequence": seq},
    )

    monkeypatch.setattr(
        cli,
        "resolve_model_entry",
        lambda spec: (_ for _ in ()).throw(
            AssertionError("local model paths should bypass resolve_model_entry")
        ),
    )
    monkeypatch.setattr(cli, "_load_impl", lambda name: fake_impl)
    monkeypatch.setattr(
        cli,
        "write_predictions",
        lambda out, records, *, structure_name: captured.update(
            {
                "out": out,
                "records": list(records),
                "structure_name": structure_name,
            }
        ),
    )

    args = cli.build_parser().parse_args(
        [
            "infer",
            "--model",
            str(model_dir),
            "--document-structure",
            "contacts-and-distances-v1",
            "--input-sequence",
            "ACD",
            "--batch-size",
            "17",
            "--out",
            str(out_path),
        ]
    )

    cli.cmd_infer(args)

    assert captured["cfg"].model == str(model_dir)
    assert captured["cfg"].batch_size == 17
    assert captured["structures"] == [{"sequence": "ACD"}]
    assert captured["out"] == out_path
    assert captured["structure_name"] == "contacts-and-distances-v1"


@dataclass(frozen=True)
class _FakeRolloutConfig(_FakeInferenceConfig):
    """An impl config that offers a readout choice, as contacts-v1 does."""

    method: str = "pairwise"
    n_rollouts: int = 100


def _fake_impl(config_cls, captured: dict[str, Any]) -> SimpleNamespace:
    def _predict(cfg, *, structures=None):
        captured["cfg"] = cfg
        yield {"entry_id": "sequence", "pairs": [], "expected_distances": []}

    return SimpleNamespace(
        NAME="fake-v1",
        InferenceConfig=config_cls,
        predict=_predict,
        structure_from_sequence=lambda seq: {"sequence": seq},
    )


def _run_infer(
    monkeypatch,
    tmp_path: Path,
    impl: SimpleNamespace,
    extra_argv: list[str],
) -> dict[str, Any]:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir(exist_ok=True)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_load_impl", lambda name: impl)
    monkeypatch.setattr(
        cli,
        "write_predictions",
        lambda out, records, *, structure_name: None,
    )
    args = cli.build_parser().parse_args(
        [
            "infer",
            "--model", str(model_dir),
            "--document-structure", "fake-v1",
            "--input-sequence", "ACD",
            "--out", str(tmp_path / "preds.json"),
            *extra_argv,
        ]
    )
    cli.cmd_infer(args)
    return captured


def test_cmd_infer_passes_rollout_flags_to_impl(monkeypatch, tmp_path: Path) -> None:
    """The README's `--method rollout --n-rollouts 100` reaches the impl."""
    captured: dict[str, Any] = {}
    impl = _fake_impl(_FakeRolloutConfig, captured)
    _run_infer(
        monkeypatch, tmp_path, impl,
        ["--method", "rollout", "--n-rollouts", "7"],
    )
    assert captured["cfg"].method == "rollout"
    assert captured["cfg"].n_rollouts == 7


def test_cmd_infer_leaves_impl_defaults_when_flags_omitted(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    impl = _fake_impl(_FakeRolloutConfig, captured)
    _run_infer(monkeypatch, tmp_path, impl, [])
    assert captured["cfg"].method == "pairwise"
    assert captured["cfg"].n_rollouts == 100


def test_cmd_infer_rejects_rollout_flags_for_impls_without_them(
    monkeypatch, tmp_path: Path
) -> None:
    """Impls whose config has no `method` field get an error, not a TypeError."""
    captured: dict[str, Any] = {}
    impl = _fake_impl(_FakeInferenceConfig, captured)
    with pytest.raises(SystemExit, match="--method is not supported"):
        _run_infer(monkeypatch, tmp_path, impl, ["--method", "rollout"])
