# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The publish manifest must still equal the identity #245 evaluated.

``publish_exp232_m2_p06.py`` carries its source URI and per-object manifest by
value, because it runs on a pod with no repo checked out. That copy is only
trustworthy while it matches the pinned identity it was copied from, so this
loads #245's ``checkpoint_specs.py`` by path — the module imports nothing from
the repo, so it can be exec'd without installing exp245 — and asserts the two
agree object for object.

    uv run --with pytest pytest test_publish_specs.py
"""

import importlib.util
from pathlib import Path

import publish_exp232_m2_p06 as publish

SPECS_PATH = (Path(__file__).resolve().parents[1]
              / "exp245_evals_foldbench_held_out_monomers/rollout/checkpoint_specs.py")


def exp245_specs():
    spec = importlib.util.spec_from_file_location("exp245_checkpoint_specs", SPECS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_uri_matches_exp245():
    checkpoint = exp245_specs().M2_P06_CHECKPOINT
    assert publish.SOURCE_URI == checkpoint.coreweave_uri
    assert publish.RUN_NAME == checkpoint.run_name
    assert publish.STEP == checkpoint.step


def test_manifest_matches_exp245():
    checkpoint = exp245_specs().M2_P06_CHECKPOINT
    pinned = {file.name: (file.size, file.digest) for file in checkpoint.files}
    assert publish.SOURCE_FILES == pinned


def test_publishing_the_other_final_would_be_caught():
    """The two #232 finals differ only in their weight ETags — pin the right ones."""
    specs = exp245_specs()
    other = {file.name: (file.size, file.digest) for file in specs.M1_P02_CHECKPOINT.files}
    assert publish.SOURCE_FILES != other
