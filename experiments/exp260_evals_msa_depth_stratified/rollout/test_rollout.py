# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused invariants for the exp260 MSA-depth evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import checkpoint_specs
import finalize_coreweave
import pandas as pd
import pytest
import run_coreweave_eval
from hf_to_s3 import expected_manifest

RESULTS = Path(__file__).parents[1] / "data" / "coreweave_results"


def test_training_suite_scores_only_the_pr257_checkpoint() -> None:
    checkpoints = checkpoint_specs.CHECKPOINT_SUITES["training"]
    assert [checkpoint.run_name for checkpoint in checkpoints] == [
        (
            "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-"
            "augcont-lr005-us-east1"
        )
    ]
    assert [checkpoint.step for checkpoint in checkpoints] == [363_000]
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
        0
    ]


def test_checkpoint_is_read_in_place_from_the_pr257_export() -> None:
    """No re-export: this run points at the HF directory PR #257 wrote."""

    checkpoint = checkpoint_specs.TRAIN_CHECKPOINT
    assert checkpoint.coreweave_uri == (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/"
        "models/exp232-decontam-train-m2-p06-step363000/hf/step-363000"
    )
    assert checkpoint.levanter_source_objects == 175
    assert checkpoint.levanter_source_bytes == 17_659_722_031
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


def test_universe_adds_eval_test_to_the_pr257_union() -> None:
    assert checkpoint_specs.EVAL_SETS == ("eval-val", "eval-test", "eval-denovo")
    assert checkpoint_specs.EXPECTED_SET_SIZES == {
        "legacy_554": 554,
        "eval-val": 97,
        "eval-test": 217,
        "eval-denovo": 19,
    }
    assert checkpoint_specs.EXPECTED_UNITS == 887
    assert sum(checkpoint_specs.EXPECTED_SET_SIZES.values()) == 887
    assert checkpoint_specs.EXPECTED_UNIQUE_STEMS == 773
    assert (
        checkpoint_specs.EXPECTED_UNITS - checkpoint_specs.EXPECTED_UNIQUE_STEMS
        == checkpoint_specs.EXPECTED_OVERLAPPING_STEMS + 2
    )


def test_scored_subsets_carry_every_reported_partition() -> None:
    manifest = pd.read_csv(RESULTS / "inputs" / "evaluation_subsets.csv")
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
    assert counts["universe_887"] == 887
    assert counts["legacy_554"] == 554
    assert counts["eval-val"] == 97
    assert counts["eval-test"] == 217
    assert counts["eval-denovo"] == 19
    assert counts["eval-val-viral"] + counts["eval-val-nonviral"] == 97
    assert counts["eval-test-viral"] + counts["eval-test-nonviral"] == 217
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


def _reference_frame(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "marinfold-exp232-decontam-train-m2-p06-step363000",
                "subset": subset,
                "range": range_name,
                "cut": cut,
                "precision": expected + offset,
            }
            for (
                subset,
                range_name,
                cut,
            ), expected in checkpoint_specs.PUBLISHED_REFERENCE_METRICS.items()
        ]
    )


def test_pr257_gate_covers_every_published_subset() -> None:
    validation = finalize_coreweave.validate_published_reference(_reference_frame(0.0))
    assert validation["applicable"]
    assert validation["passed"]
    assert len(validation["comparisons"]) == 12
    assert {record["subset"] for record in validation["comparisons"]} == {
        "legacy_554",
        "eval-val",
        "eval-denovo",
    }


def test_pr257_gate_fails_outside_tolerance() -> None:
    validation = finalize_coreweave.validate_published_reference(_reference_frame(0.02))
    assert not validation["passed"]
    assert validation["largest_absolute_difference"] == pytest.approx(0.02)


def test_pr257_gate_is_inapplicable_without_the_checkpoint() -> None:
    frame = _reference_frame(0.0)
    frame["model"] = "marinfold-e8-reference-step35679"
    validation = finalize_coreweave.validate_published_reference(frame)
    assert not validation["applicable"]
    assert validation["passed"]


def test_worker_matches_the_pr244_accepted_rollout_implementation() -> None:
    worker = Path(run_coreweave_eval.__file__).with_name("score_rollout_worker.py")
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == (
        finalize_coreweave.PR244_ACCEPTED_WORKER_SHA256
    )


def test_run_root_is_dated_and_isolated() -> None:
    assert checkpoint_specs.run_root("v1-01") == (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp260_evals_msa_depth_stratified/rollout-v2/2026-08-31/v1-01"
    )
    with pytest.raises(ValueError, match="invalid run id"):
        checkpoint_specs.run_root("bad/run")
