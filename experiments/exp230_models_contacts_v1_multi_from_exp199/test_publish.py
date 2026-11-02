# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""The rope repair, pinned against the real configs.

levanter's HF export writes rope in the transformers-5 form only. transformers
4.x does not read ``rope_parameters``: it silently falls back to a default rope
and loses 0.76 nats/token with no error. The repair has to reproduce EXACTLY
what a 4.x-readable checkpoint carries, and the ground truth for "exactly" is
exp199's own config, which has all three keys.

Values below are copied from the real files on the training node.
"""
from __future__ import annotations

import pytest

from publish_to_hf_bucket import repair_rope

ROPE_PARAMS = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 8192, "rope_type": "llama3",
               "rope_theta": 500000}
# exp199's config.json, which IS 4.x-readable -- the target the repair must hit.
BASE_ROPE_THETA = 500000
BASE_ROPE_SCALING = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"}


def test_repair_reproduces_the_base_config_exactly():
    """The exported config, repaired, must match exp199's rope keys key-for-key."""
    exported = {"rope_theta": None, "rope_scaling": None, "rope_parameters": dict(ROPE_PARAMS)}
    fixed, notes = repair_rope(exported)
    assert fixed["rope_theta"] == BASE_ROPE_THETA
    assert fixed["rope_scaling"] == BASE_ROPE_SCALING
    # rope_scaling is rope_parameters WITHOUT rope_theta -- not a superset, not a
    # subset with extras. A stray rope_theta inside rope_scaling is what 4.x
    # chokes on.
    assert "rope_theta" not in fixed["rope_scaling"]
    assert len(notes) == 2


def test_repair_is_idempotent():
    """Running it on an already-4.x config must change nothing."""
    already = {"rope_theta": BASE_ROPE_THETA, "rope_scaling": dict(BASE_ROPE_SCALING),
               "rope_parameters": dict(ROPE_PARAMS)}
    fixed, notes = repair_rope(already)
    assert notes == []
    assert fixed["rope_theta"] == BASE_ROPE_THETA
    assert fixed["rope_scaling"] == BASE_ROPE_SCALING


def test_repair_leaves_a_config_without_rope_parameters_alone():
    plain = {"rope_theta": 10000}
    fixed, notes = repair_rope(plain)
    assert fixed == {"rope_theta": 10000} and notes == []


def test_tokenizer_check_rejects_the_wrong_tokenizer(tmp_path):
    """Publishing exp199's tokenizer with an exp230 checkpoint must be fatal."""
    import json
    from publish_to_hf_bucket import check_tokenizer

    def write(id7, n=2845):
        vocab = {f"<t{i}>": i for i in range(n)}
        # id 7 is renamed IN PLACE, so the name at 7 is what identifies the mode.
        del vocab["<t7>"]
        vocab[id7] = 7
        (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}}))

    write("<contacts-and-distances-v1>")               # exp199's, the wrong one
    with pytest.raises(SystemExit, match="expected"):
        check_tokenizer(tmp_path)

    write("<contacts-v1.multi>")                       # exp230's
    check_tokenizer(tmp_path)                          # must not raise

    write("<contacts-v1.multi>", n=2846)               # a resize means id drift
    with pytest.raises(SystemExit, match="vocab"):
        check_tokenizer(tmp_path)


def test_missing_tokenizer_is_fatal(tmp_path):
    from publish_to_hf_bucket import check_tokenizer
    with pytest.raises(SystemExit, match="no tokenizer.json"):
        check_tokenizer(tmp_path)
