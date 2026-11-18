"""Reject unrelated resume state and ignore incomplete checkpoint writes."""

import copy
import json

import fsspec

from tpu_checkpoint import (
    ROOT,
    committed_checkpoints,
    frozen_source,
    validate_provenance,
)


def test_checkpoint_selection_and_provenance() -> None:
    """Resume the latest committed numeric step with all original logical cursors."""
    snapshot = {
        "fork_run": "test-exp347-recovery",
        "parent_run": "original-gpu-run",
        "parent": {"step": 9, "tokens": 100},
        "format": "contacts_v1",
    }
    metadata = {
        "source_snapshot": frozen_source(snapshot),
        "source_run": snapshot["parent_run"],
        "format": "contacts_v1",
        "world": 8,
        "target_tokens": 100_000_000_000,
        "step": 11,
        "tokens": 180,
        "cursors": [{"rank": rank, "world": 8} for rank in range(8)],
    }
    validate_provenance(metadata, snapshot)
    invalid = [
        {**metadata, "source_run": "unrelated"},
        {**metadata, "format": "prompted"},
        {**metadata, "target_tokens": 200_000_000_000},
        {**metadata, "cursors": metadata["cursors"][:-1]},
        {**metadata, "step": 8},
    ]
    swapped = copy.deepcopy(metadata)
    swapped["cursors"][0], swapped["cursors"][1] = (
        swapped["cursors"][1],
        swapped["cursors"][0],
    )
    invalid.append(swapped)
    for candidate in invalid:
        try:
            validate_provenance(candidate, snapshot)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted an invalid continuation")
    fs = fsspec.filesystem("memory")
    root = f"{ROOT}/checkpoints/{snapshot['fork_run']}"
    for step in (9, 10, 11):
        fs.pipe_file(
            f"{root}/step-{step}/complete.json",
            json.dumps({**metadata, "step": step}).encode(),
        )
    fs.pipe_file(f"{root}/step-12/state/incomplete", b"interrupted write")
    fs.pipe_file(
        f"{root}/converted/step-9/complete.json",
        json.dumps({**metadata, "step": 9}).encode(),
    )
    assert committed_checkpoints(fs, root) == [
        f"{root}/step-{step}" for step in (9, 10, 11)
    ]
