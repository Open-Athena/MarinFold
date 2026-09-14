# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused invariants for the exp277 rollout-v2 evaluation."""

import base64
import hashlib
import json
from pathlib import Path

import checkpoint_specs
import run_coreweave_eval
from hf_to_s3 import expected_manifest


def test_checkpoint_identity_and_manifest_are_pinned() -> None:
    (checkpoint,) = checkpoint_specs.CHECKPOINT_SUITES["exp277"]
    assert checkpoint.run_name == "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
    assert checkpoint.step == 266_344
    assert checkpoint.source_dtype == "float32"
    assert checkpoint.coreweave_uri.startswith(
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
    )
    files = {file.name: file for file in checkpoint.files}
    assert set(files) == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert sum(file.size for file in files.values()) == 5_885_614_887
    assert all(file.digest_kind == "s3-etag" for file in files.values())


def test_eval_universe_excludes_eval_test() -> None:
    assert checkpoint_specs.EVAL_SETS == ("eval-val", "eval-denovo")
    assert checkpoint_specs.EXPECTED_SET_SIZES == {
        "legacy_554": 554,
        "eval-val": 97,
        "eval-denovo": 19,
    }
    assert checkpoint_specs.EXPECTED_UNITS == 670
    assert checkpoint_specs.EXPECTED_UNIQUE_STEMS == 556


def test_worker_and_rollout_recipe_match_validated_path() -> None:
    worker = Path(run_coreweave_eval.__file__).with_name("score_rollout_worker.py")
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == (
        "dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c"
    )
    checkpoint = checkpoint_specs.EXP277_CHECKPOINT
    manifest = base64.b64encode(
        json.dumps(expected_manifest(checkpoint), sort_keys=True).encode()
    ).decode()
    command = run_coreweave_eval._child_command(
        worker_b64="worker",
        model_manifest_b64=manifest,
        model_uri=checkpoint.coreweave_uri,
        targets_uri="s3://marin-us-east-02a/MarinFold/targets.parquet",
        output_uri="s3://marin-us-east-02a/MarinFold/output",
        label=checkpoint.label,
        shard_idx=0,
        num_shards=12,
        seed=0,
        contact_mult=6,
        accept_unfinished=True,
    )
    shell = command[-1]
    for expected in (
        "--n-rollouts 100",
        "--temperature 1.0",
        "--top-p 0.95",
        "--top-k -1",
        "--contact-mult 6",
        "--shard 0/12",
        "--accept-unfinished",
    ):
        assert expected in shell
    assert "s.bind((\"\",0))" in shell


def test_output_root_is_experiment_scoped() -> None:
    assert checkpoint_specs.run_root("v2-01") == (
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
        "evals/rollout-v2/2026-09-13/v2-01"
    )
