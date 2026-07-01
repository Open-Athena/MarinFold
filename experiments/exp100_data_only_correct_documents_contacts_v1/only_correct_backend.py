# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM V1 **structured-output backend** for only-correct contacts-v1 decoding.

This is the portable masking path (issue #100): vLLM's TPU stack (marin
`tpu_inference`) does **not** invoke custom `LogitsProcessor.apply()`, but it
*does* apply the structured-output **grammar bitmask** on-device in JAX
(`structured_decoding_manager.structured_decode_fn` sets disallowed tokens to
`-inf` before sampling). So we express the only-correct constraint as a
`StructuredOutputGrammar` whose `fill_bitmask` emits the per-step allowed-token
bitmask from our FSM (`constrained_grammar.OnlyCorrectMatcher`). The same code
runs on GPU vLLM (develop/debug here, in-process) and on iris TPU (register via a
vLLM general-plugin entry point).

Backend selection in V1 is a hardcoded if/elif in `StructuredOutputManager`
(no plugin hook) and the manager runs in the EngineCore process, so :func:`register`
monkeypatches (a) `Processor._validate_structured_output` to accept our backend +
skip the xgrammar-syntax validation, and (b) `StructuredOutputManager.grammar_init`
to build :class:`OnlyCorrectBackend`. For an in-process GPU engine set
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` before importing vllm so the patches take
effect; for multiprocess/TPU, call :func:`register` from a `vllm.general_plugins`
entry point (loaded in every process).

Unmodified NLL is recovered separately via a `prompt_logprobs` pass (supported on
this tpu_inference rev; computed from raw pre-mask logits) — the bitmask only
handles masking.
"""
from __future__ import annotations

import json

import torch

from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

from constrained_grammar import OnlyCorrectMatcher, pack_allowed_bitmask

BACKEND_NAME = "onlycorrect"


def make_grammar_spec(gt_pos_ids, contact_id: int, end_id: int) -> str:
    """Per-request grammar spec string (goes in ``StructuredOutputsParams.grammar``)."""
    return json.dumps({
        "gt": [[int(a), int(b)] for a, b in gt_pos_ids],
        "contact_id": int(contact_id),
        "end_id": int(end_id),
    })


def is_our_spec(grammar: str | None) -> bool:
    if not isinstance(grammar, str):
        return False
    try:
        d = json.loads(grammar)
        return isinstance(d, dict) and "gt" in d and "contact_id" in d
    except (ValueError, TypeError):
        return False


class OnlyCorrectGrammar(StructuredOutputGrammar):
    """Request-level grammar: wraps the incremental FSM + emits its bitmask."""

    def __init__(self, matcher: OnlyCorrectMatcher, vocab_size: int):
        self.m = matcher
        self.vocab_size = vocab_size
        self.n_words = (vocab_size + 31) // 32
        self.history: list[int] = []

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        for t in tokens:
            if not self.m.accept(int(t)):
                return False
            self.history.append(int(t))
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        # accepted prefix from current state, without advancing (fresh replay)
        probe = OnlyCorrectMatcher(
            [tuple(p) for p in self.m._initial],  # frozenset iterates to 2 ids
            contact_id=self.m.contact_id, end_id=self.m.end_id)
        for t in self.history:
            probe.accept(t)
        out: list[int] = []
        for t in tokens:
            if probe.accept(int(t)):
                out.append(int(t))
            else:
                break
        return out

    def rollback(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        self.history = self.history[:-num_tokens]
        self.m.reset()
        for t in self.history:
            self.m.accept(t)

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        row = pack_allowed_bitmask(self.m.allowed(), self.n_words)
        bitmask[idx] = torch.from_numpy(row).to(bitmask.dtype)

    def is_terminated(self) -> bool:
        return self.m.terminated

    def reset(self) -> None:
        self.m.reset()
        self.history.clear()


class OnlyCorrectBackend(StructuredOutputBackend):
    """Engine-level backend (dataclass fields vllm_config, tokenizer, vocab_size)."""

    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str
                        ) -> StructuredOutputGrammar:
        spec = json.loads(grammar_spec)
        matcher = OnlyCorrectMatcher([tuple(p) for p in spec["gt"]],
                                     contact_id=spec["contact_id"], end_id=spec["end_id"])
        return OnlyCorrectGrammar(matcher, self.vocab_size)

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        # int32 [max_num_seqs, ceil(vocab/32)], initialized all-allowed (-1 == all bits set)
        return torch.full((max_num_seqs, (self.vocab_size + 31) // 32), -1, dtype=torch.int32)

    def destroy(self) -> None:
        pass


_REGISTERED = False


def register() -> None:
    """Monkeypatch vLLM V1 to route our backend. Idempotent. Call before requests
    (and in every engine process — see module docstring)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.v1.engine.processor import Processor
    from vllm.v1 import structured_output as so

    _orig_validate = Processor._validate_structured_output

    def _patched_validate(self, params):
        sp = getattr(params, "structured_outputs", None)
        if sp is not None and self.structured_outputs_config and is_our_spec(sp.grammar):
            if self.model_config.skip_tokenizer_init:
                raise ValueError("structured outputs need a tokenizer")
            sp._backend = BACKEND_NAME  # bypass xgrammar/guidance syntax validation
            return
        return _orig_validate(self, params)

    Processor._validate_structured_output = _patched_validate

    _orig_init = so.StructuredOutputManager.grammar_init

    def _patched_grammar_init(self, request):
        if request.structured_output_request is None:
            return
        sp = request.sampling_params
        if sp is not None and sp.structured_outputs is not None \
                and getattr(sp.structured_outputs, "_backend", None) == BACKEND_NAME:
            if self.backend is None:
                vocab = self.vllm_config.model_config.get_vocab_size()
                self.backend = OnlyCorrectBackend(
                    self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab)
            grammar = self.executor.submit(self._async_create_grammar, request)
            request.structured_output_request.grammar = grammar
            return
        return _orig_init(self, request)

    so.StructuredOutputManager.grammar_init = _patched_grammar_init
    _REGISTERED = True
