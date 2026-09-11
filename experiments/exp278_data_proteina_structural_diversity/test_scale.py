"""Protect scale-run identities and recovery without requiring an accelerator."""

import io
import json
from pathlib import Path

import fsspec
import numpy as np
import pytest

from launch import create_worker_request
from scale_common import paused, sampling_history
from scale_plan import make_plan
from scale_snapshot import case_counts


def test_backbones_recover_without_aggregate_timing_or_progress_files(
    tmp_path: Path,
) -> None:
    original = [
        {"batch_index": 3, "elapsed_seconds": 1.2, "sample_in_batch": i}
        for i in range(2)
    ]
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer, ca=np.zeros((2, 60, 3)), timings_json=json.dumps(original)
    )
    (tmp_path / "batch-00003.npz").write_bytes(buffer.getvalue())
    rows, indices = sampling_history(fsspec.filesystem("file"), str(tmp_path), 60, 2)
    assert rows == original and indices == {3}
    assert not (tmp_path / "progress.json").exists()


def test_misplaced_archive_cannot_silently_skip_a_different_batch(
    tmp_path: Path,
) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer, ca=np.zeros((1, 60, 3)), timings_json=json.dumps([{"batch_index": 9}])
    )
    (tmp_path / "batch-00003.npz").write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="name differs"):
        sampling_history(fsspec.filesystem("file"), str(tmp_path), 60, 1)


def test_scale_plan_has_disjoint_seeds_and_balanced_independent_queues() -> None:
    plan = make_plan("test", 768, 1_000_000)
    seeds = [
        case["seed"] + batch
        for case in plan["cases"]
        for batch in range(case["batches"])
    ]
    assert len(seeds) == len(set(seeds))
    assert max(seeds) < 2**32
    loads = [
        sum(
            case["estimated_gpu_seconds"]
            for case in plan["cases"]
            if case["worker"] == worker
        )
        for worker in range(768)
    ]
    assert min(loads) > 0
    assert max(loads) - min(loads) < max(
        case["estimated_gpu_seconds"] for case in plan["cases"]
    )
    assert {case["length"] for case in plan["cases"]} == set(range(60, 501))


def test_scale_jobs_are_singletons_at_batch_priority() -> None:
    request = create_worker_request(
        "test",
        ["scale_worker.py"],
        "s3://bucket/code.tgz",
        "abc",
        "short,long",
        100,
        preemption_retries=100,
    )
    assert request.priority == 3
    assert request.replicas == 1
    assert request.processes_per_task == 1
    assert request.max_retries_preemption == 100


def test_continue_review_does_not_pause_and_explicit_pause_does(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"pause": False, "pause_at_utc": None}))
    assert not paused(str(path))
    path.write_text(json.dumps({"pause": True, "pause_at_utc": None}))
    assert paused(str(path))


def test_snapshot_counts_saved_sequences_before_refold_commit(tmp_path: Path) -> None:
    root = str(tmp_path)
    case = {
        "id": "case1",
        "length": 60,
        "condition": "unconditional",
        "worker": 0,
        "batch_size": 2,
        "batches": 1,
    }
    paths = {root + "/cases/case1/generated/batch-00000.npz"}
    result = case_counts(case, root, paths, fsspec.filesystem("file"), {"case1": 1})
    assert result["generated"] == result["sequences_saved"] == 2
    assert result["refolded"] == result["quality_pass"] == result["complete"] == 0
