# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused invariants for the exp232 rollout-v2 evaluation."""

import base64
import json

import pandas as pd
import pytest

import checkpoint_specs
import finalize_coreweave
import run_coreweave_eval
from hf_to_s3 import expected_manifest


def test_suite_is_exactly_the_three_approved_in_place_checkpoints() -> None:
    checkpoints = checkpoint_specs.CHECKPOINT_SUITES["exp232"]
    assert [checkpoint.run_name for checkpoint in checkpoints] == [
        "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084",
        "prot-exp232-cw-cv1-decontam-s02-m2-p06-aug",
        "prot-exp232-cw-cv1-decontam-s02-m1-p02-aug",
    ]
    assert [checkpoint.step for checkpoint in checkpoints] == [35_679, 145_199, 145_199]
    assert all(
        checkpoint.coreweave_uri.startswith("s3://marin-us-east-02a/")
        for checkpoint in checkpoints
    )
    assert all(
        "/hf/step-145199" in checkpoint.coreweave_uri for checkpoint in checkpoints[1:]
    )
    assert all(
        checkpoint_specs.checkpoint_model_uri("unused", checkpoint)
        == checkpoint.coreweave_uri
        for checkpoint in checkpoints
    )


def test_exp232_exports_include_weights_and_tokenizer() -> None:
    for checkpoint in checkpoint_specs.CHECKPOINTS[1:]:
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


def test_only_m1_p02_accepts_the_observed_seven_unfinished_rollouts() -> None:
    checkpoints = checkpoint_specs.CHECKPOINT_SUITES["exp232"]
    assert [checkpoint.accepted_unfinished_rollouts for checkpoint in checkpoints] == [
        0,
        0,
        7,
    ]


def test_child_command_preserves_rollout_recipe() -> None:
    checkpoint = checkpoint_specs.M2_P06_CHECKPOINT
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
        accept_unfinished=True,
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
        "--accept-unfinished",
    ]:
        assert fragment in shell


def test_gcs_is_rejected() -> None:
    checkpoint = checkpoint_specs.M2_P06_CHECKPOINT
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


def test_e8_gate_runs_when_reference_is_in_exp232_suite() -> None:
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
    validation = finalize_coreweave.validate_e8_reference(pd.DataFrame(rows), "exp232")
    assert validation is not None
    assert validation["passed"]
    assert len(validation["comparisons"]) == 4


def test_run_root_is_isolated_and_validated() -> None:
    assert checkpoint_specs.run_root("decontam-v2-20260818-01") == (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp232_sweep_cv1_decontam/evals/rollout_v2/decontam-v2-20260818-01"
    )
    with pytest.raises(ValueError, match="invalid run id"):
        checkpoint_specs.run_root("bad/run")
