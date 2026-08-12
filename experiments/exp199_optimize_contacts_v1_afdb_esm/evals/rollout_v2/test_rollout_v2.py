# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Static identity and request-shape tests for the CoreWeave eval."""

import base64
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

import run_coreweave_eval
import submit_coreweave
from checkpoint_specs import (
    CHECKPOINT_SUITES,
    CHECKPOINTS,
    CONTINUATION_CHECKPOINT,
    CONTINUATION_HF_REVISION,
    E8_HF_REVISION,
    E8_REFERENCE_CHECKPOINT,
    E8_REFERENCE_METRICS,
    EXP199_HF_REVISION,
    MARIN_PREFIX,
    checkpoint_model_uri,
    model_s3_uri,
    run_root,
)
from finalize_coreweave import validate_e8_reference
from hf_to_s3 import expected_manifest, hf_file_url


def test_checkpoint_manifests_are_immutable_and_complete() -> None:
    assert len(CHECKPOINTS) == 3
    assert len({checkpoint.run_name for checkpoint in CHECKPOINTS}) == 3
    for checkpoint in CHECKPOINTS:
        manifest = expected_manifest(checkpoint)
        assert len(manifest["files"]) == 6
        assert sum(file["size"] for file in manifest["files"]) == 5_885_614_712
        assert all(
            EXP199_HF_REVISION in hf_file_url(checkpoint, file)
            for file in checkpoint.files
        )
        assert all(
            file.digest_kind in {"sha256", "git-sha1"} for file in checkpoint.files
        )


def test_continuation_checkpoint_is_an_isolated_pinned_suite() -> None:
    checkpoint = CONTINUATION_CHECKPOINT
    assert CHECKPOINT_SUITES["continuation"] == (checkpoint,)
    assert CHECKPOINT_SUITES["exp199"] == CHECKPOINTS
    assert checkpoint.hf_revision == CONTINUATION_HF_REVISION
    assert checkpoint.run_name == (
        "prot-exp199-cv1-cont-s03-m1-p03-srcbase-aug100-us-east1"
    )
    assert checkpoint.step == 145_199
    assert checkpoint.weight_shard_digests == (
        "3aaa2198ccf399813da9d68a8d08355354be4f7b60eced074bcc8209d69b0cf0",
        "bbfaad34189c5292970255395e1566f4f9a6f83b528085e7ff1b3b58dcfd9d34",
    )
    manifest = expected_manifest(checkpoint)
    assert len(manifest["files"]) == 6
    assert sum(file["size"] for file in manifest["files"]) == 5_885_614_712
    assert all(
        CONTINUATION_HF_REVISION in hf_file_url(checkpoint, file)
        for file in checkpoint.files
    )
    assert model_s3_uri("contbase-v2-20260812-01", checkpoint) == (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/"
        "contbase-v2-20260812-01/models/"
        "prot-exp199-cv1-cont-s03-m1-p03-srcbase-aug100-us-east1/hf/step-145199"
    )


def test_run_paths_are_coreweave_s3_only() -> None:
    root = run_root("v2-test-01")
    assert root.startswith(f"{MARIN_PREFIX}/")
    assert "gs://" not in root
    for checkpoint in CHECKPOINTS:
        assert model_s3_uri("v2-test-01", checkpoint).startswith(f"{root}/models/")
    assert checkpoint_model_uri("v2-test-01", E8_REFERENCE_CHECKPOINT) == (
        "s3://marin-us-east-02a/MarinFold/exp163/model/step-35679"
    )


def test_e8_reference_is_the_exact_existing_coreweave_hf_export() -> None:
    checkpoint = E8_REFERENCE_CHECKPOINT
    manifest = expected_manifest(checkpoint)
    assert manifest["source"] == {
        "repo_id": "open-athena/marinfold-exp75",
        "revision": E8_HF_REVISION,
        "subfolder": ("prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679"),
    }
    assert len(manifest["files"]) == 7
    assert sum(file["size"] for file in manifest["files"]) == 5_885_616_184
    assert checkpoint.weight_shard_digests == (
        "0be51806a5ecbcbd4a7e2824c2c687a56e4bf0d5861db40a6432714270ccf50a",
        "67cf32f6959292aaea53de2082d83f39af87a829237660fdbc74ce9af960451e",
    )


def test_e8_reference_acceptance_gate() -> None:
    rows = [
        {
            "model": "marinfold-e8-reference-step35679",
            "range": range_name,
            "cut": cut,
            "precision": expected,
        }
        for (range_name, cut), expected in E8_REFERENCE_METRICS.items()
    ]
    passing = validate_e8_reference(pd.DataFrame(rows), "e8-reference")
    assert passing is not None and passing["passed"]

    rows[0]["precision"] += 0.0051
    failing = validate_e8_reference(pd.DataFrame(rows), "e8-reference")
    assert failing is not None and not failing["passed"]


def test_checked_in_e8_evidence_matches_checkpoint_spec() -> None:
    data_directory = Path(__file__).with_name("data")
    verification = json.loads(
        (data_directory / "e8_checkpoint_verification.json").read_text()
    )
    validation = json.loads(
        (data_directory / "e8_reference_validation.json").read_text()
    )
    checkpoint = E8_REFERENCE_CHECKPOINT
    assert verification["coreweave_uri"] == checkpoint.coreweave_uri
    assert verification["source"] == expected_manifest(checkpoint)["source"]
    assert verification["files"] == expected_manifest(checkpoint)["files"]
    assert validation["passed"]
    assert (
        max(
            comparison["absolute_difference"]
            for comparison in validation["comparisons"]
        )
        < 0.005
    )


def test_checked_in_continuation_verification_matches_checkpoint_spec() -> None:
    verification = json.loads(
        (
            Path(__file__).with_name("data")
            / "continuation_checkpoint_verification.json"
        ).read_text()
    )
    checkpoint = CONTINUATION_CHECKPOINT
    assert verification["verified_from_huggingface"]
    assert verification["source"] == expected_manifest(checkpoint)["source"]
    assert verification["files"] == expected_manifest(checkpoint)["files"]
    assert verification["destination"] == model_s3_uri(
        "contbase-v2-20260812-01", checkpoint
    )


def test_checked_in_timings_cover_all_five_evaluated_checkpoints() -> None:
    timings = pd.read_csv(Path(__file__).with_name("data") / "timings.csv")
    assert len(timings) == 5 * 554
    assert set(timings.model_nickname) == {
        "trc_p03_aug_step72599",
        "trc_p03_base_step72599",
        "cw_p06_aug_step145199",
        "trc_cont_srcbase_aug100_step145199",
        "e8_reference_step35679",
    }
    assert timings.complete.all()
    assert timings.unfinished_rollouts.sum() == 0
    assert set(timings.n_rollouts) == {100}


def test_child_request_has_fixed_recipe_and_batch_h100_shape() -> None:
    checkpoint = CHECKPOINTS[0]
    command = run_coreweave_eval._child_command(
        worker_b64="cHJpbnQoJ29rJykK",
        model_manifest_b64=base64.b64encode(
            json.dumps(expected_manifest(checkpoint)).encode()
        ).decode(),
        model_uri=model_s3_uri("v2-test-01", checkpoint),
        targets_uri=f"{run_root('v2-test-01')}/inputs/eval_targets.parquet",
        output_uri=f"{run_root('v2-test-01')}/rollout",
        label=checkpoint.label,
        shard_idx=3,
        num_shards=12,
        vllm_port=20_300,
        seed=7,
    )
    shell = command[2]
    for expected in (
        "--shard 3/12",
        "--n-rollouts 100",
        "--temperature 1.0",
        "--top-p 0.95",
        "--top-k -1",
        "export VLLM_PORT=20300",
        "--seed 7",
        MARIN_PREFIX,
    ):
        assert expected in shell
    assert "gs://" not in shell

    request = run_coreweave_eval._job_request(name="exp199-test", command=command)
    request_data = dataclasses.asdict(request)
    assert request.priority == run_coreweave_eval.IRIS_PRIORITY_BAND_BATCH
    assert request.replicas == 1
    assert request_data["resources"]["device"]["variant"] == "H100"
    assert request_data["resources"]["device"]["count"] == 1
    assert request.environment.env_vars["MARIN_PREFIX"] == MARIN_PREFIX


def test_root_submission_uses_marin_cluster_and_eczech_user(monkeypatch) -> None:
    observed = {}

    def capture(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs

    monkeypatch.setattr(sys, "argv", ["submit_coreweave.py"])
    monkeypatch.setattr(submit_coreweave.subprocess, "run", capture)
    submit_coreweave.main()

    command = observed["command"]
    assert "--cluster=marin" in command
    assert command[command.index("--user") + 1] == "eczech"
    assert command[command.index("--target-cluster") + 1] == "cw-us-east-02a"
