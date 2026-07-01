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

    def __init__(self, gt_pairs, contact_id: int, end_id: int, vocab_size: int):
        self._gt = [tuple(p) for p in gt_pairs]
        self.contact_id = contact_id
        self.end_id = end_id
        self.m = OnlyCorrectMatcher(self._gt, contact_id=contact_id, end_id=end_id)
        self.vocab_size = vocab_size
        self.n_words = (vocab_size + 31) // 32
        self.history: list[int] = []
        self.broken = False  # set if a token slips past the mask (state unrecoverable)

    def _new_matcher(self) -> OnlyCorrectMatcher:
        return OnlyCorrectMatcher(self._gt, contact_id=self.contact_id, end_id=self.end_id)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        # History-authoritative + tolerant: NEVER return False (a False makes vLLM
        # terminate the request). The mask is applied correctly by vLLM; the only
        # failure mode we've seen is our *incremental* state momentarily drifting
        # from the true token stream under the engine's fill/accept scheduling. On
        # any mismatch we resync by replaying the authoritative history; only if the
        # replayed stream is genuinely illegal (mask truly failed) do we mark the
        # rollout broken (it just won't be selected — it isn't 100%-correct).
        for t in tokens:
            t = int(t)
            if self.broken:
                self.history.append(t)
                continue
            if self.m.accept(t):
                self.history.append(t)
                continue
            probe = self._new_matcher()
            if all(probe.accept(x) for x in self.history) and probe.accept(t):
                self.m = probe                      # resynced from history
                self.history.append(t)
            else:
                self.broken = True                  # genuine mask miss; abandon constraint
                self.history.append(t)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        probe = self._new_matcher()
        for t in self.history:
            probe.accept(t)
        out: list[int] = []
        for t in tokens:
            if not self.broken and probe.accept(int(t)):
                out.append(int(t))
            else:
                break
        return out

    def rollback(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        self.history = self.history[:-num_tokens]
        self.m = self._new_matcher()
        self.broken = False
        for t in self.history:
            if not self.m.accept(t):
                self.broken = True
                break

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # broken -> full mask (no constraint); the rollout is already non-correct.
        allowed = range(self.vocab_size) if self.broken else self.m.allowed()
        row = pack_allowed_bitmask(allowed, self.n_words)
        bitmask[idx] = torch.from_numpy(row).to(bitmask.dtype)

    def is_terminated(self) -> bool:
        return self.m.terminated and not self.broken

    def reset(self) -> None:
        self.m = self._new_matcher()
        self.history.clear()
        self.broken = False


class OnlyCorrectBackend(StructuredOutputBackend):
    """Engine-level backend (dataclass fields vllm_config, tokenizer, vocab_size)."""

    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str
                        ) -> StructuredOutputGrammar:
        spec = json.loads(grammar_spec)
        return OnlyCorrectGrammar([tuple(p) for p in spec["gt"]],
                                  contact_id=spec["contact_id"], end_id=spec["end_id"],
                                  vocab_size=self.vocab_size)

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        # int32 [max_num_seqs, ceil(vocab/32)], initialized all-allowed (-1 == all bits set)
        return torch.full((max_num_seqs, (self.vocab_size + 31) // 32), -1, dtype=torch.int32)

    def destroy(self) -> None:
        pass


_REGISTERED = False


def _sop_backend(sampling_params):
    so = getattr(sampling_params, "structured_outputs", None)
    return getattr(so, "_backend", None) if so is not None else None


def register() -> None:
    """Monkeypatch vLLM V1 to route our only-correct backend. Idempotent, and
    version-robust across the vLLM layout differences we've hit:

    * grammar_init: we *pre-set* ``self.backend`` to OnlyCorrectBackend for our
      requests, then delegate to the original — both the 0.11 and 0.20 forks skip
      their hardcoded if/elif dispatch when ``self.backend`` is already set, so we
      don't have to reimplement their (differing) grammar-creation tail.
    * validation: 0.20 validates in ``SamplingParams._validate_structured_outputs``
      (called via ``params.verify``); 0.11 validated in
      ``Processor._validate_structured_output``. We patch whichever exists so the
      xgrammar/guidance syntax check is skipped for our spec and ``_backend`` is
      set to ours.

    Must run in every engine process (see module docstring)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.v1 import structured_output as so

    # (1) grammar_init: pre-set our backend, then delegate.
    _orig_init = so.StructuredOutputManager.grammar_init

    def _patched_grammar_init(self, request):
        if request.structured_output_request is not None \
                and _sop_backend(request.sampling_params) == BACKEND_NAME \
                and self.backend is None:
            vocab = self.vllm_config.model_config.get_vocab_size()
            self.backend = OnlyCorrectBackend(
                self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab)
        return _orig_init(self, request)

    so.StructuredOutputManager.grammar_init = _patched_grammar_init

    # (2) validation: skip syntax check + set _backend for our spec.
    patched = False
    try:  # vLLM 0.20.x — SamplingParams._validate_structured_outputs(cfg, tokenizer)
        from vllm.sampling_params import SamplingParams
        _orig_v = SamplingParams._validate_structured_outputs

        def _patched_v(self, structured_outputs_config, tokenizer):
            sp = getattr(self, "structured_outputs", None)
            if sp is not None and structured_outputs_config is not None and is_our_spec(sp.grammar):
                if tokenizer is None:
                    raise ValueError("structured outputs need a tokenizer")
                sp._backend = BACKEND_NAME
                return
            return _orig_v(self, structured_outputs_config, tokenizer)

        SamplingParams._validate_structured_outputs = _patched_v
        patched = True
    except (ImportError, AttributeError):
        pass
    if not patched:  # vLLM 0.11.x — Processor._validate_structured_output(params)
        from vllm.v1.engine.processor import Processor
        _orig_p = Processor._validate_structured_output

        def _patched_p(self, params):
            sp = getattr(params, "structured_outputs", None)
            if sp is not None and self.structured_outputs_config and is_our_spec(sp.grammar):
                if self.model_config.skip_tokenizer_init:
                    raise ValueError("structured outputs need a tokenizer")
                sp._backend = BACKEND_NAME
                return
            return _orig_p(self, params)

        Processor._validate_structured_output = _patched_p

    _REGISTERED = True
