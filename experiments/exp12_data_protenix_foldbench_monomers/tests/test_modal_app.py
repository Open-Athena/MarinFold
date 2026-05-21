# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modal_app import _inject_precomputed_msa_paths, _seed_outputs_complete  # noqa: E402


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "_scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job_data() -> list[dict]:
    return [{
        "name": "5sbj_A",
        "sequences": [{"proteinChain": {"sequence": "ACD", "count": 1}}],
        "covalent_bonds": [],
    }]


def test_seed_outputs_complete_requires_every_sample(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_1"
    seed_dir.mkdir()
    (seed_dir / "5sbj_A_distogram.npz").write_bytes(b"npz")
    (seed_dir / "5sbj_A_sample_0.cif").write_text("sample 0")
    (seed_dir / "5sbj_A_summary_confidence_sample_0.json").write_text("{}")

    assert not _seed_outputs_complete(seed_dir, stem="5sbj_A", n_sample=2)

    (seed_dir / "5sbj_A_sample_1.cif").write_text("sample 1")
    (seed_dir / "5sbj_A_summary_confidence_sample_1.json").write_text("{}")

    assert _seed_outputs_complete(seed_dir, stem="5sbj_A", n_sample=2)


def test_inject_precomputed_msa_paths_warns_without_files(tmp_path: Path, capsys) -> None:
    job_data = _job_data()

    found_precomputed_msa = _inject_precomputed_msa_paths(
        job_data,
        stem="5sbj_A",
        msa_root=tmp_path,
    )

    assert not found_precomputed_msa
    assert "auto-search at inference time" in capsys.readouterr().out
    protein_chain = job_data[0]["sequences"][0]["proteinChain"]
    assert "pairedMsaPath" not in protein_chain
    assert "unpairedMsaPath" not in protein_chain


def test_inject_precomputed_msa_paths_uses_existing_files(tmp_path: Path) -> None:
    msa_base = tmp_path / "5sbj_A" / "msa" / "0"
    pairing = msa_base / "pairing.a3m"
    non_pairing = msa_base / "0" / "non_pairing.a3m"
    non_pairing.parent.mkdir(parents=True)
    pairing.write_text(">query\nACD\n")
    non_pairing.write_text(">query\nACD\n")

    job_data = _job_data()

    found_precomputed_msa = _inject_precomputed_msa_paths(
        job_data,
        stem="5sbj_A",
        msa_root=tmp_path,
    )

    assert found_precomputed_msa
    protein_chain = job_data[0]["sequences"][0]["proteinChain"]
    assert protein_chain["pairedMsaPath"] == str(pairing)
    assert protein_chain["unpairedMsaPath"] == str(non_pairing)


def test_dispatch_timing_requires_cached_msas() -> None:
    dispatch_timing = _load_script("dispatch_timing")

    with pytest.raises(SystemExit, match="missing cached MSA"):
        dispatch_timing._require_cached_msas([
            {"stem": "5sbj_A", "paired_exists": True, "non_pairing_exists": True},
            {"stem": "7t9r_A", "paired_exists": False, "non_pairing_exists": True},
        ])


def test_dispatch_timing_wait_for_results_raises_on_failures(capsys) -> None:
    dispatch_timing = _load_script("dispatch_timing")

    class _DummyFuture:
        def __init__(self, *, result=None, error: Exception | None = None) -> None:
            self._result = result
            self._error = error

        def get(self):
            if self._error is not None:
                raise self._error
            return self._result

    futures = [
        _DummyFuture(result={"elapsed_seconds": 1.0, "model_load_seconds": 2.0}),
        _DummyFuture(error=RuntimeError("boom")),
    ]
    call_args = [
        {"mode": "single_seq", "stem": "5sbj_A"},
        {"mode": "msa", "stem": "7t9r_A"},
    ]

    with pytest.raises(SystemExit, match="timing jobs failed"):
        dispatch_timing._wait_for_results(futures, call_args)

    out = capsys.readouterr().out
    assert "[done] single_seq/5sbj_A" in out
    assert "[FAIL] msa/7t9r_A: boom" in out


def test_collect_timings_raises_on_modal_failure(monkeypatch, tmp_path: Path) -> None:
    collect_timings = _load_script("collect_timings")

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="volume missing",
        )

    monkeypatch.setattr(collect_timings.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="modal volume get failed for msa/5sbj_A"):
        collect_timings.sync_timings_json("msa", "5sbj_A", tmp_path)
