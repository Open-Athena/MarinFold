# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Changed, incomplete or wrong-count data fails before training starts."""

import json

import pytest

from experiments.exp279_models_exact_soft_contact_targets.inputs import (
    inspect_cache,
    source_identity,
    verify_manifest,
)
from experiments.exp279_models_exact_soft_contact_targets.recipe import CORPORA


def test_frozen_inputs_reject_changes(tmp_path):
    inputs = {}
    for name, (_, rows, tokens) in CORPORA.items():
        path = tmp_path / name
        path.mkdir()
        ledger = {
            "is_finished": True,
            "total_num_rows": rows or 1,
            "field_counts": {"input_ids": tokens or 64},
        }
        (path / "shard_ledger.json").write_text(json.dumps(ledger))
        inputs[name] = inspect_cache(str(path), name)
    manifest = {"source": source_identity(), "inputs": inputs}
    verify_manifest(manifest)
    manifest["source"]["code_sha256"] = "changed"
    with pytest.raises(ValueError, match="Source"):
        verify_manifest(manifest)
    manifest["source"] = source_identity()
    path = tmp_path / "val" / "shard_ledger.json"
    ledger = json.loads(path.read_text())
    ledger["field_counts"]["input_ids"] += 1
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(manifest)


def test_wrong_or_incomplete_training_cache_is_rejected(tmp_path):
    path = tmp_path / "shard_ledger.json"
    path.write_text(json.dumps({"is_finished": False}))
    with pytest.raises(ValueError, match="Incomplete"):
        inspect_cache(str(tmp_path), "afdb")
    path.write_text(
        json.dumps(
            {"is_finished": True, "total_num_rows": 1, "field_counts": {"input_ids": 1}}
        )
    )
    with pytest.raises(ValueError, match="counts differ"):
        inspect_cache(str(tmp_path), "afdb")
