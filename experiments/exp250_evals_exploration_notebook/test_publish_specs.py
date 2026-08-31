# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Each publish manifest must still equal the identity its evaluation pinned.

``publish_exp232_m2_p06.py`` carries its source URIs and per-object manifests by
value, because it runs on a pod with no repo checked out. Those copies are only
trustworthy while they match the pinned identities they were copied from, so this
loads the two ``checkpoint_specs.py`` by path — neither imports anything from the
repo, so they can be exec'd without installing the experiments — and asserts they
agree object for object.

    uv run --with pytest pytest test_publish_specs.py
"""

import importlib.util
import sys
from pathlib import Path

import publish_exp232_m2_p06 as publish

EXPERIMENTS = Path(__file__).resolve().parents[1]
#: evaluation -> the specs module that pinned a checkpoint published here.
SPEC_PATHS = {
    "exp245": EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/rollout/checkpoint_specs.py",
    "exp232_0824": (EXPERIMENTS / "exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2"
                    / "checkpoint_specs.py"),
}


def specs(name):
    path = SPEC_PATHS[name]
    module_name = f"{name}_checkpoint_specs"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: `dataclasses` resolves a class's module out of
    # sys.modules while building __init__, and a spec module that defines
    # dataclasses fails with a bare AttributeError without this.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def manifest(checkpoint):
    """The pinned manifest as this script states it: name -> (size, digest)."""
    files = getattr(checkpoint, "files", None) or checkpoint.checkpoint_files
    return {file.name: (file.size, file.digest) for file in files}


def test_sweep_matches_exp245():
    pinned = specs("exp245").M2_P06_CHECKPOINT
    published = publish.CHECKPOINTS["sweep"]
    assert published.source_uri == pinned.coreweave_uri
    assert published.run_name == pinned.run_name
    assert published.step == pinned.step
    assert published.files == manifest(pinned)


def test_training_matches_exp232_august_evaluation():
    pinned = specs("exp232_0824").TRAIN_CHECKPOINT
    published = publish.CHECKPOINTS["training"]
    assert published.source_uri == pinned.coreweave_uri
    assert published.run_name == pinned.run_name
    assert published.step == pinned.step
    assert published.files == manifest(pinned)


def test_the_two_finals_are_not_confusable():
    """The #232 finals differ only in their weight ETags — pin the right ones."""
    other = manifest(specs("exp245").M1_P02_CHECKPOINT)
    assert publish.CHECKPOINTS["sweep"].files != other
    assert publish.CHECKPOINTS["training"].files != other
    assert (publish.CHECKPOINTS["training"].files
            != publish.CHECKPOINTS["sweep"].files)


def test_the_default_is_the_better_checkpoint():
    assert publish.DEFAULT_CHECKPOINT == "training"
    assert publish.SPEC is publish.CHECKPOINTS["training"]


def test_bucket_paths_are_per_run():
    paths = {key: publish.bucket_path(spec) for key, spec in publish.CHECKPOINTS.items()}
    assert len(set(paths.values())) == len(paths)
    for key, path in paths.items():
        assert path.startswith("checkpoints/") and path.endswith(
            f"/hf/step-{publish.CHECKPOINTS[key].step}")
