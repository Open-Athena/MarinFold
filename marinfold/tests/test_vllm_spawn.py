# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The vLLM backend must not let its EngineCore be forked.

vLLM's default multiproc method is fork, and forking a parent that has
already done native work deadlocks the child inside `_init_executor` —
silently, with the GPU idle. contacts-v1 `evaluate` triggers it every
time: it computes pyconfind ground truth before building the backend.
"""

import os

import pytest

pytest.importorskip("vllm")

from marinfold.inference import _vllm  # noqa: E402


def _construct(monkeypatch, calls: list) -> None:
    """Build a VllmBackend with vLLM's LLM stubbed out."""
    class _StubLLM:
        def __init__(self, **kwargs):
            calls.append(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"))

        def get_tokenizer(self):
            return object()

    monkeypatch.setattr(_vllm, "LLM", _StubLLM)
    monkeypatch.setattr(_vllm, "model_source_path", lambda p: str(p))
    _vllm.VllmBackend("/stub")


def test_engine_defaults_to_spawn(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    seen: list = []
    _construct(monkeypatch, seen)
    # Set before LLM() is constructed, not after — the fork happens in there.
    assert seen == ["spawn"]


def test_explicit_setting_is_left_alone(monkeypatch, tmp_path):
    """An operator who asked for fork gets fork; we only fill in a default."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    seen: list = []
    _construct(monkeypatch, seen)
    assert seen == ["fork"]
