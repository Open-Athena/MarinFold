# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise bundle provenance and production phase/resource boundaries."""

import json
import os
import subprocess
import sys

import pytest

from experiments.exp279_models_exact_soft_contact_targets.inputs import source_identity
from experiments.exp279_models_exact_soft_contact_targets.launch import (
    stage_bundle,
    worker_request,
)
from experiments.exp279_models_exact_soft_contact_targets.train import phase_for_update


def test_phase_boundaries_and_completed_run():
    expected = {
        0: "base",
        217800: "base",
        217801: "recovery",
        333960: "recovery",
        333961: "final",
        363000: "final",
        363001: None,
    }
    for update, phase in expected.items():
        assert phase_for_update(update) == phase
    for update in (-1, 363002):
        with pytest.raises(ValueError, match="outside"):
            phase_for_update(update)


def test_worker_bundle_reproduces_source_identity_without_git(tmp_path):
    identity = source_identity()
    stage_bundle(tmp_path, {"source": identity})
    assert not (tmp_path / ".git").exists()
    env = {**os.environ, "EXP279_GIT_SHA": identity["git_sha"], "JAX_PLATFORMS": "cpu"}
    command = [
        sys.executable,
        "-c",
        "import json; from experiments.exp279_models_exact_soft_contact_targets.inputs import source_identity; print(json.dumps(source_identity()))",
    ]
    copied = json.loads(
        subprocess.check_output(command, cwd=tmp_path, env=env, text=True)
    )
    assert copied == identity
    source = tmp_path / "experiments/exp279_models_exact_soft_contact_targets/model.py"
    source.write_text(source.read_text() + "\n# deliberate bundle corruption\n")
    corrupted = json.loads(
        subprocess.check_output(command, cwd=tmp_path, env=env, text=True)
    )
    assert corrupted["code_sha256"] != identity["code_sha256"]


def test_tpu_gang_requests_all_hosts_and_preserves_runtime_contract():
    request = worker_request(
        name="test",
        arm="soft",
        run_name="exp279-soft-test",
        output="gs://marin-us-east1/protein-structure/MarinFold/exp279",
        region="us-east1",
        tpu="v6e-32",
        stop_after=32,
        env={},
    )
    assert request.replicas == 8
    assert request.resources.device.chip_count() == 4
    assert request.resources.zone == "us-east1-d"
    assert request.resources.regions == ["us-east1"]
    assert request.resources.preemptible
    assert request.priority == 3
    command = request.entrypoint.binary_entrypoint.args[-1]
    assert "--locked" in command and "--extra tpu" in command
    assert "--stop-after 32" in command and "--resume-latest" in command
    assert request.environment.setup_scripts == []
