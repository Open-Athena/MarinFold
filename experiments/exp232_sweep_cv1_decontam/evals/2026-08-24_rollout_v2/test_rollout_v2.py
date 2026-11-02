# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused invariants for the 2026-08-24 exp232 rollout-v2 evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import checkpoint_specs
import export_training_checkpoint
import finalize_coreweave
import pandas as pd
import pytest
import run_coreweave_eval
from hf_to_s3 import expected_manifest


def test_suite_contains_only_validation_and_selected_training_checkpoint() -> None:
    checkpoints = checkpoint_specs.CHECKPOINT_SUITES["training"]
    assert [checkpoint.run_name for checkpoint in checkpoints] == [
        "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084",
        (
            "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-"
            "augcont-lr005-us-east1"
        ),
    ]
    assert [checkpoint.step for checkpoint in checkpoints] == [35_679, 363_000]
    assert all(
        checkpoint.coreweave_uri.startswith("s3://marin-us-east-02a/")
        for checkpoint in checkpoints
    )
    assert all(
        checkpoint_specs.checkpoint_model_uri("unused", checkpoint)
        == checkpoint.coreweave_uri
        for checkpoint in checkpoints
    )
    assert [checkpoint.accepted_unfinished_rollouts for checkpoint in checkpoints] == [
        0,
        0,
    ]


def test_training_source_and_eval_local_export_are_pinned() -> None:
    checkpoint = checkpoint_specs.TRAIN_CHECKPOINT
    assert checkpoint.levanter_source_objects == 175
    assert checkpoint.levanter_source_bytes == 17_659_722_031
    assert checkpoint.levanter_source_manifest_sha256 == (
        "7e0c6f650fe6c76a5570695c24e447ddad4ae0def6662371c17b3ad1fd656b37"
    )
    assert (
        checkpoint.levanter_source_uri == export_training_checkpoint.SOURCE_CHECKPOINT
    )
    assert checkpoint.coreweave_uri == export_training_checkpoint.OUTPUT_CHECKPOINT

    files = {file.name: file for file in checkpoint.files}
    assert set(files) == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert sum(file.size for file in files.values()) == 5_885_614_712
    assert all(file.digest_kind == "s3-etag" for file in files.values())


def test_current_sets_exclude_eval_test_and_retain_legacy_comparison() -> None:
    assert checkpoint_specs.EVAL_SETS == ("eval-val", "eval-denovo")
    assert checkpoint_specs.EXPECTED_SET_SIZES == {
        "legacy_554": 554,
        "eval-val": 97,
        "eval-denovo": 19,
    }
    assert checkpoint_specs.EXPECTED_UNITS == 670


def test_current_eval_sets_retain_viral_partitions() -> None:
    manifest = pd.read_csv(
        Path(__file__).parent / "data" / "coreweave_results" / "evaluation_subsets.csv"
    )
    precision = manifest[["dataset", "stem"]].copy()
    precision["model"] = "synthetic"
    precision["range"] = "all"
    precision["cut"] = "R"
    precision["precision"] = 1.0
    aggregate, counts = finalize_coreweave.aggregate_subsets(
        precision,
        ordered_units=list(zip(manifest.dataset, manifest.stem, strict=True)),
        subset_manifest=manifest,
    )
    assert counts == {
        "universe_670": 670,
        "legacy_554": 554,
        "eval-val": 97,
        "eval-denovo": 19,
        "eval-val-nonviral": 91,
        "eval-val-viral": 6,
        "eval-denovo-nonviral": 19,
    }
    assert set(aggregate.subset) == set(counts)


def test_child_command_preserves_rollout_recipe() -> None:
    checkpoint = checkpoint_specs.TRAIN_CHECKPOINT
    manifest = base64.b64encode(
        json.dumps(expected_manifest(checkpoint), sort_keys=True).encode()
    ).decode()
    command = run_coreweave_eval._child_command(
        worker_b64="worker",
        model_manifest_b64=manifest,
        model_uri=checkpoint.coreweave_uri,
        targets_uri="s3://marin-us-east-02a/marin/targets.parquet",
        output_uri="s3://marin-us-east-02a/marin/output",
        label=checkpoint.label,
        shard_idx=0,
        num_shards=12,
        vllm_port=20_000,
        seed=0,
        contact_mult=6,
        accept_unfinished=False,
    )
    shell = command[-1]
    for fragment in [
        "--n-rollouts 100",
        "--temperature 1.0",
        "--top-p 0.95",
        "--top-k -1",
        "--contact-mult 6",
        "--shard 0/12",
        "--seed 0",
    ]:
        assert fragment in shell
    assert "--accept-unfinished" not in shell


def test_gcs_is_rejected() -> None:
    checkpoint = checkpoint_specs.TRAIN_CHECKPOINT
    with pytest.raises(ValueError, match="GCS sources are forbidden"):
        run_coreweave_eval._child_command(
            worker_b64="worker",
            model_manifest_b64="manifest",
            model_uri=checkpoint.coreweave_uri,
            targets_uri="gs://forbidden/targets.parquet",
            output_uri="s3://marin-us-east-02a/marin/output",
            label=checkpoint.label,
            shard_idx=0,
            num_shards=12,
            vllm_port=20_000,
            seed=0,
            contact_mult=6,
            accept_unfinished=False,
        )


def test_e8_gate_checks_all_four_legacy_headlines() -> None:
    model = "marinfold-e8-reference-step35679"
    rows = [
        {
            "model": model,
            "subset": "legacy_554",
            "range": range_name,
            "cut": cut,
            "precision": expected,
        }
        for (range_name, cut), expected in checkpoint_specs.E8_REFERENCE_METRICS.items()
    ]
    validation = finalize_coreweave.validate_e8_reference(
        pd.DataFrame(rows), "training"
    )
    assert validation is not None
    assert validation["passed"]
    assert len(validation["comparisons"]) == 4


def test_worker_matches_the_pr244_accepted_rollout_implementation() -> None:
    worker = Path(run_coreweave_eval.__file__).with_name("score_rollout_worker.py")
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == (
        finalize_coreweave.PR244_ACCEPTED_WORKER_SHA256
    )


def test_run_root_is_dated_and_isolated() -> None:
    assert checkpoint_specs.run_root("v2-01") == (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01"
    )
    with pytest.raises(ValueError, match="invalid run id"):
        checkpoint_specs.run_root("bad/run")
