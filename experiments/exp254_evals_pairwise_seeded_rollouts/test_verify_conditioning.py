# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from verify_conditioning import ARMS, PAYLOAD_SUFFIXES, position_frames, sha256, verify


def seal_unit(run: Path, plan_path: Path, stem: str = "unit") -> None:
    """Write a completion manifest after constructing or deliberately altering a fixture."""
    payloads = {
        suffix: (run / "units" / f"{stem}.{suffix}").read_bytes()
        for suffix in PAYLOAD_SUFFIXES
    }
    marker = {
        "stem": stem,
        "plan_sha256": sha256(plan_path.read_bytes()),
        "elapsed_seconds": 20.0,
        "files": {
            suffix: {"bytes": len(payload), "sha256": sha256(payload)}
            for suffix, payload in payloads.items()
        },
    }
    (run / "units" / f"{stem}.complete.json").write_text(json.dumps(marker))


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[Path, Path]:
    contexts = {
        arm: []
        if arm in ("iid", "iid_repeat")
        else [[2, 8]]
        if arm.startswith("false")
        else [[0, 6]]
        for arm in ARMS
    }
    target = {
        "dataset": "test",
        "stem": "unit",
        "L": 38,
        "input_seq": "A" * 38,
        "resolved": list(range(38)),
        "truth": [[0, 6], [1, 7]],
        "contexts": [contexts, contexts],
    }
    source_path = tmp_path / "source_votes.npz"
    np.savez(source_path, unit=np.zeros((38, 38), dtype=np.int16))
    plan = {
        "arms": ARMS,
        "n_repeats": 2,
        "n_rollouts": 2,
        "source_n_rollouts": 2,
        "model_run": "test-model",
        "step": 123,
        "max_model_len": 8192,
        "source_votes_file": source_path.name,
        "source_votes_sha256": sha256(source_path.read_bytes()),
        "targets": [target],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    run = tmp_path / "run"
    (run / "units").mkdir(parents=True)
    frames = position_frames(target, 2, 2)
    raw, arrays, rows = {}, {}, []
    for repeat in range(2):
        for arm in ARMS:
            key = f"r{repeat}__{arm}"
            raw[key] = []
            for positions in frames[repeat]:
                # The final mention repeats a contact in reverse orientation;
                # both the stored list and votes must count it only once.
                text = (
                    f"<contact> <p{positions[0]}> <p{positions[6]}> "
                    f"<contact> <p{positions[1]}> <p{positions[7]}> "
                    f"<contact> <p{positions[6]}> <p{positions[0]}>"
                )
                raw[key].append(
                    {
                        "text": text,
                        "contacts": [[0, 6], [1, 7]],
                        "finish_reason": "stop",
                        "tokens": 10,
                    }
                )
            votes = np.zeros((38, 38), dtype=np.int16)
            for i, j in ((0, 6), (1, 7)):
                votes[i, j] = votes[j, i] = 2
            arrays[key + "__votes"] = votes
            if arm != "iid_repeat":
                arrays[key + "__prob"] = votes.astype(float) / 10
            copied = 2 if contexts[arm] == [[0, 6]] else 0
            rows.append(
                {
                    "stem": "unit",
                    "n_residues": 38,
                    "n_pairs": 703,
                    "mode": arm,
                    "replicate": repeat,
                    "n_rollouts": 2,
                    "n_given": len(contexts[arm]),
                    "elapsed_seconds": 0.5,
                    "probability_probe_seconds": 0.0 if arm == "iid_repeat" else 0.1,
                    "total_seconds": 0.7,
                    "prompt_tokens": 100,
                    "max_new_tokens": 356,
                    "generated_tokens": 20,
                    "copied_context_pairs": copied,
                    "novel_generated_pairs": 4 - copied,
                    "unfinished_rollouts": 0,
                    "timestamp_utc": "2026-09-09T12:00:00+00:00",
                    "model_nickname": "test-model-step-123",
                    "model_source": "/test/model",
                    "model_load_seconds": 2.0,
                    "runner_tag": "local",
                    "gpu_name": "test GPU",
                    "gpu_total_memory_gb": 24.0,
                    "gpu_compute_capability": "8.0",
                    "hostname": "test-host",
                    "platform": "Linux",
                    "torch_version": "2.0",
                    "vllm_version": "0.19.1",
                    "transformers_version": "5.15.0",
                    "plan_sha256": sha256(plan_path.read_bytes()),
                    "worker_sha256": "a" * 64,
                }
            )
    np.savez(run / "units" / "unit.npz", **arrays)
    (run / "units" / "unit.raw.json.gz").write_bytes(
        gzip.compress(json.dumps(raw).encode(), mtime=0)
    )
    pd.DataFrame(rows).to_csv(run / "units" / "unit.timings.csv", index=False)
    seal_unit(run, plan_path)
    return plan_path, run


def test_complete_raw_roundtrip_and_vote_reconstruction(completed_run):
    plan, run = completed_run
    report = verify(plan, run)
    assert report["scope"] == "full_frozen_plan"
    assert report["n_verified_completions"] == 32
    assert report["units"][0]["n_groups"] == 16
    assert report["units"][0]["empty_rollouts"] == 0


def test_checksum_detects_payload_corruption(completed_run):
    plan, run = completed_run
    path = run / "units" / "unit.raw.json.gz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum or length"):
        verify(plan, run)


def test_resealed_wrong_contacts_fail_independent_raw_parse(completed_run):
    plan, run = completed_run
    path = run / "units" / "unit.raw.json.gz"
    raw = json.loads(gzip.decompress(path.read_bytes()))
    raw["r0__iid"][0]["contacts"] = [[0, 6]]
    path.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    seal_unit(run, plan)
    with pytest.raises(ValueError, match="differ from reparsed raw text"):
        verify(plan, run)


def test_resealed_wrong_votes_fail_even_when_saved_contact_lists_are_right(
    completed_run,
):
    plan, run = completed_run
    path = run / "units" / "unit.npz"
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["r0__iid__votes"][0, 6] = arrays["r0__iid__votes"][6, 0] = 1
    np.savez(path, **arrays)
    seal_unit(run, plan)
    with pytest.raises(ValueError, match="saved votes differ"):
        verify(plan, run)


def test_resealed_extra_group_is_rejected(completed_run):
    plan, run = completed_run
    path = run / "units" / "unit.raw.json.gz"
    raw = json.loads(gzip.decompress(path.read_bytes()))
    raw["r99__iid"] = raw["r0__iid"]
    path.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    seal_unit(run, plan)
    with pytest.raises(ValueError, match="missing or extra groups"):
        verify(plan, run)


def test_full_run_rejects_extra_units_while_smoke_is_explicitly_scoped(completed_run):
    plan, run = completed_run
    (run / "units" / "extra.complete.json").write_text("{}")
    with pytest.raises(ValueError, match="missing/extra unit files"):
        verify(plan, run)
    smoke = verify(plan, run, stem="unit")
    assert smoke["scope"] == "operational_smoke"
    assert "accuracy" not in smoke


def test_full_run_rejects_missing_payload(completed_run):
    plan, run = completed_run
    (run / "units" / "unit.npz").unlink()
    with pytest.raises(ValueError, match="missing/extra unit files"):
        verify(plan, run)


def test_resealed_incorrect_telemetry_cannot_pass_on_hashes_alone(completed_run):
    plan, run = completed_run
    path = run / "units" / "unit.timings.csv"
    timings = pd.read_csv(path)
    timings.loc[0, "copied_context_pairs"] = 1
    timings.to_csv(path, index=False)
    seal_unit(run, plan)
    with pytest.raises(ValueError, match="copied_context_pairs disagrees"):
        verify(plan, run)


def set_fixture_termination(plan: Path, run: Path, reason: str, tokens: int) -> None:
    """Change one termination and reconcile its telemetry, preserving actual contacts."""
    path = run / "units" / "unit.raw.json.gz"
    raw = json.loads(gzip.decompress(path.read_bytes()))
    raw["r0__false_large"][0]["finish_reason"] = reason
    raw["r0__false_large"][0]["tokens"] = tokens
    path.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    timing_path = run / "units" / "unit.timings.csv"
    timings = pd.read_csv(timing_path)
    selected = (timings.replicate == 0) & (timings["mode"] == "false_large")
    timings.loc[selected, "generated_tokens"] = tokens + 10
    timings.loc[selected, "unfinished_rollouts"] = int(reason != "stop")
    timings.to_csv(timing_path, index=False)
    seal_unit(run, plan)


def test_budget_termination_requires_explicit_opt_in(completed_run):
    plan, run = completed_run
    set_fixture_termination(plan, run, "length", 356)
    with pytest.raises(ValueError, match="requires explicit acceptance"):
        verify(plan, run)


def test_opted_in_exact_budget_completion_is_retained_and_counted(completed_run):
    plan, run = completed_run
    set_fixture_termination(plan, run, "length", 356)
    result = verify(plan, run, accept_budget_termination=True)
    assert result["n_verified_completions"] == 32
    assert result["budget_terminated_rollouts"] == 1
    assert result["budget_terminated_by_arm"]["false_large"] == 1
    assert result["units"][0]["budget_terminated_by_group"]["r0__false_large"] == 1


@pytest.mark.parametrize("tokens", [355, 357])
def test_opt_in_does_not_accept_false_budget_termination(completed_run, tokens):
    plan, run = completed_run
    set_fixture_termination(plan, run, "length", tokens)
    with pytest.raises(ValueError, match="token (budget|count)"):
        verify(plan, run, accept_budget_termination=True)


def test_opt_in_does_not_accept_unknown_finish_reason(completed_run):
    plan, run = completed_run
    set_fixture_termination(plan, run, "error", 356)
    with pytest.raises(ValueError, match="unknown finish reason"):
        verify(plan, run, accept_budget_termination=True)


def test_explicit_worker_allowlist_rejects_unknown_digest(completed_run):
    plan, run = completed_run
    with pytest.raises(ValueError, match="outside the explicit allowlist"):
        verify(plan, run, worker_sha256=["b" * 64])


def add_second_worker_unit(plan_path: Path, run: Path) -> None:
    """Construct a second target with independently rebuilt position-token text."""
    plan = json.loads(plan_path.read_text())
    second = copy.deepcopy(plan["targets"][0])
    second["stem"] = "unit_b"
    plan["targets"].append(second)
    source_path = plan_path.parent / plan["source_votes_file"]
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    source["unit_b"] = source["unit"].copy()
    np.savez(source_path, **source)
    plan["source_votes_sha256"] = sha256(source_path.read_bytes())
    plan_path.write_text(json.dumps(plan))
    frames = position_frames(second, plan["n_repeats"], plan["n_rollouts"])
    raw = json.loads(gzip.decompress((run / "units" / "unit.raw.json.gz").read_bytes()))
    for repeat in range(plan["n_repeats"]):
        for arm in ARMS:
            for completion, positions in zip(
                raw[f"r{repeat}__{arm}"], frames[repeat], strict=True
            ):
                completion["text"] = " ".join(
                    f"<contact> <p{positions[i]}> <p{positions[j]}>"
                    for i, j in ((0, 6), (1, 7), (6, 0))
                )
    (run / "units" / "unit_b.raw.json.gz").write_bytes(
        gzip.compress(json.dumps(raw).encode(), mtime=0)
    )
    (run / "units" / "unit_b.npz").write_bytes(
        (run / "units" / "unit.npz").read_bytes()
    )
    timings = pd.read_csv(run / "units" / "unit.timings.csv")
    timings["plan_sha256"] = sha256(plan_path.read_bytes())
    timings.to_csv(run / "units" / "unit.timings.csv", index=False)
    timings["stem"] = "unit_b"
    timings["worker_sha256"] = "b" * 64
    timings.to_csv(run / "units" / "unit_b.timings.csv", index=False)
    seal_unit(run, plan_path)
    seal_unit(run, plan_path, "unit_b")


def test_default_rejects_two_workers_and_explicit_allowlist_accepts_them(completed_run):
    plan, run = completed_run
    add_second_worker_unit(plan, run)
    with pytest.raises(ValueError, match="without an explicit allowlist"):
        verify(plan, run)
    result = verify(plan, run, worker_sha256=["a" * 64, "b" * 64])
    assert result["n_verified_targets"] == 2
    assert result["observed_worker_sha256"] == ["a" * 64, "b" * 64]


def test_worker_allowlist_does_not_relax_gpu_identity(completed_run):
    plan, run = completed_run
    add_second_worker_unit(plan, run)
    path = run / "units" / "unit_b.timings.csv"
    timings = pd.read_csv(path)
    timings["gpu_name"] = "different GPU"
    timings.to_csv(path, index=False)
    seal_unit(run, plan, "unit_b")
    with pytest.raises(ValueError, match="incompatible worker metadata: gpu_name"):
        verify(plan, run, worker_sha256=["a" * 64, "b" * 64])
