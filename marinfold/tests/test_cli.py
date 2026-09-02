# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
