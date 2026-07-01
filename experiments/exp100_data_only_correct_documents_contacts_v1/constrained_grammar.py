# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Constrained ("only-correct") decoding for contacts-v1 structure sections.

The contacts-v1 structure section is a flat list of ``<contact> <pi> <pj>``
statements terminated by ``<end>`` (see the document_structures/contacts_v1
SPEC). The tokenizer is **WordLevel** (whitespace-separated, strictly 1:1), so a
statement is exactly the three-token stream ``[<contact>, <pi>, <pj>]`` — there
are no inter-token space tokens. That makes the grammar a clean 3-token cycle.

This module enforces, at every generation step, that the model can *only* emit a
token that keeps the rollout on a path toward a fully-correct document (issue
#100): it may only complete a **true** contact that has not yet been emitted, and
``<end>`` is masked until **every** true contact has been emitted (so the
finished document has recall == precision == 1.0 by construction). The only
freedom left to the model is the *order* of the true contacts and which endpoint
of each pair it writes first.

Two layers:

* :func:`legal_token_ids` — a **pure** function ``generated_ids -> set[int]`` of
  the token ids allowed next. No model, no torch — unit-testable in isolation.
* :class:`ContactConstraint` — a stateful, per-rollout vLLM logits-processor
  (``(past_token_ids, logits) -> logits``). vLLM is inconsistent across
  versions/backends about whether ``past_token_ids`` includes the prompt; the
  adapter **auto-detects** the prompt offset on the first call (by checking
  whether ``past_token_ids`` starts with the known prefix ids) and slices it off,
  so it operates purely on the generated suffix either way.

Contacts are held in **position-token-id space**: the caller converts each GT
seq-index pair ``(i, j)`` to the realization's position tokens via
``seq_positions`` (``pos = (n_term_index + i) % 2000``; see gen_prompts.py) and
then to token ids via the tokenizer. Orientation is symmetric — a true contact
``{a, b}`` may be emitted as either ``<contact> <a> <b>`` or ``<contact> <b> <a>``.
"""
from __future__ import annotations

from collections.abc import Iterable


def build_remaining(gt_pos_pairs: Iterable[tuple[int, int]]) -> set[frozenset[int]]:
    """The set of true contacts as orientation-free ``frozenset({a, b})`` pairs.

    ``a`` / ``b`` are **position-token ids** (already mapped from seq indices and
    encoded by the tokenizer). Self-pairs are dropped defensively.
    """
    out: set[frozenset[int]] = set()
    for a, b in gt_pos_pairs:
        if a != b:
            out.add(frozenset((a, b)))
    return out


def _replay(generated_ids: list[int], contact_id: int,
            remaining: set[frozenset[int]]) -> tuple[set[frozenset[int]], int | None]:
    """Walk completed 3-token statements, returning (still-remaining, pending_pi).

    ``pending_pi`` is the position id chosen as the first endpoint of an
    *in-progress* statement (when ``len % 3 == 2``), else ``None``. Raises
    ValueError if the stream violates the grammar — that should never happen
    under our own mask, so it surfaces a logic bug rather than passing silently.
    """
    rem = set(remaining)
    n = len(generated_ids)
    n_complete = n // 3
    for k in range(n_complete):
        c, pi, pj = generated_ids[3 * k:3 * k + 3]
        if c != contact_id:
            raise ValueError(f"statement {k} does not start with <contact> (got {c})")
        pair = frozenset((pi, pj))
        if pair not in rem:
            raise ValueError(f"statement {k} emitted non-remaining contact {tuple(pair)}")
        rem.discard(pair)
    pending_pi = generated_ids[-1] if (n % 3) == 2 else None
    return rem, pending_pi


def legal_token_ids(generated_ids: list[int], *, contact_id: int, end_id: int,
                    remaining_full: set[frozenset[int]]) -> set[int]:
    """Token ids allowed as the next token, given the generated suffix so far.

    * cycle position 0 (statement start): ``{<contact>}`` if contacts remain,
      else ``{<end>}`` (forces full recall, then stop).
    * cycle position 1 (first endpoint): every position id that is an endpoint of
      some still-remaining contact.
    * cycle position 2 (second endpoint): every position id that completes a
      still-remaining contact with the pending first endpoint.
    """
    remaining, pending_pi = _replay(generated_ids, contact_id, remaining_full)
    phase = len(generated_ids) % 3
    if phase == 0:
        return {contact_id} if remaining else {end_id}
    if phase == 1:
        return {pid for pair in remaining for pid in pair}
    # phase == 2: complete the pending contact
    return {next(iter(pair - {pending_pi}))
            for pair in remaining if pending_pi in pair}


class OnlyCorrectMatcher:
    """Incremental (O(1)/token) only-correct FSM for the structured-output path.

    Same grammar as :func:`legal_token_ids` but maintains state incrementally
    instead of replaying the whole prefix each step — needed because vLLM calls
    ``fill_bitmask`` / ``accept_tokens`` every decode step (replay would be
    O(n^2)/rollout). Cross-checked against :func:`legal_token_ids` in the tests.

    ``phase`` is the position in the 3-token ``<contact> <pi> <pj>`` cycle.
    """

    def __init__(self, gt_pos_pairs: Iterable[tuple[int, int]], *, contact_id: int, end_id: int):
        self.contact_id = contact_id
        self.end_id = end_id
        self._initial = build_remaining(gt_pos_pairs)
        self.reset()

    def reset(self) -> None:
        self.remaining = set(self._initial)
        self.phase = 0
        self.pending_pi: int | None = None
        self.terminated = False

    def allowed(self) -> set[int]:
        if self.terminated:
            return set()
        if self.phase == 0:
            return {self.contact_id} if self.remaining else {self.end_id}
        if self.phase == 1:
            return {pid for pair in self.remaining for pid in pair}
        return {next(iter(pair - {self.pending_pi}))
                for pair in self.remaining if self.pending_pi in pair}

    def accept(self, token: int) -> bool:
        """Advance by one realized token. Returns False (no-op) if not allowed."""
        if token not in self.allowed():
            return False
        if self.phase == 0:
            if token == self.end_id:
                self.terminated = True
            else:  # <contact>
                self.phase = 1
        elif self.phase == 1:
            self.pending_pi = token
            self.phase = 2
        else:  # phase 2: complete the contact
            self.remaining.discard(frozenset((self.pending_pi, token)))
            self.pending_pi = None
            self.phase = 0
        return True


def pack_allowed_bitmask(allowed: Iterable[int], n_words: int):
    """Pack an allowed-token-id set into an int32 ``[n_words]`` vLLM bitmask row.

    Convention (matches xgrammar / the TPU ``structured_decode_fn``): bit set to 1
    means the token is **allowed**; 0 means masked to ``-inf``. Returns a numpy
    int32 array (build with uint32 then view as int32 to place bit 31 safely).
    """
    import numpy as np
    row = np.zeros(n_words, dtype=np.uint32)
    for t in allowed:
        row[t >> 5] |= np.uint32(1) << np.uint32(t & 31)
    return row.view(np.int32)


class ContactConstraint:
    """Per-rollout vLLM logits processor enforcing only-correct contacts **and**
    capturing the rollout's *unmodified* (full-vocabulary) NLL.

    One instance per rollout (closes over that rollout's prefix + GT contacts).
    Pass via ``SamplingParams(logits_processors=[ContactConstraint(...)])`` with a
    *per-prompt* SamplingParams so each rollout gets its own state.

    **Why NLL is captured here** rather than in a second ``prompt_logprobs`` pass:
    ``prompt_logprobs`` returns ``None`` on the iris JAX/TPU stack (it bit exp89),
    so a separate scoring pass is not portable. Instead, at each step we read the
    incoming logits *before* masking — that is the unmodified next-token
    distribution — take its log-softmax, and lazily record the log-prob of the
    token that actually gets realized (seen on the next call). The final token
    (``<end>``) is recorded by :meth:`finalize` after generation, since the
    processor is not called again once it is emitted. ``struct_nll`` (the issue's
    "original likelihood, full output vocabulary") is then ``-sum(token_logprobs)``
    over the generated structure section.
    """

    def __init__(self, prompt_token_ids: list[int], gt_pos_pairs: Iterable[tuple[int, int]],
                 *, contact_id: int, end_id: int, capture_nll: bool = True):
        self.prompt_token_ids = list(prompt_token_ids)
        self.contact_id = contact_id
        self.end_id = end_id
        self.remaining_full = build_remaining(gt_pos_pairs)
        self.capture_nll = capture_nll
        self.token_logprobs: list[float] = []  # unmodified logprob of each realized gen token
        self._offset: int | None = None  # resolved on first call
        self._prev_logrow = None          # log-softmax of the previous step's (pre-mask) logits
        self._prev_len: int | None = None

    def _generated(self, past_token_ids: list[int]) -> list[int]:
        if self._offset is None:
            # Auto-detect whether vLLM prepends the prompt to past_token_ids.
            n = len(self.prompt_token_ids)
            self._offset = n if list(past_token_ids[:n]) == self.prompt_token_ids else 0
        return list(past_token_ids[self._offset:])

    def __call__(self, past_token_ids, logits):
        import torch  # local import: keep the pure FSM importable without torch
        gen = self._generated(past_token_ids)
        # Record the unmodified logprob of the token realized since the last call
        # (contiguity guard: only when exactly one new token appeared).
        if (self.capture_nll and self._prev_logrow is not None
                and len(gen) == self._prev_len + 1):
            self.token_logprobs.append(float(self._prev_logrow[gen[-1]]))
        if self.capture_nll:
            self._prev_logrow = torch.log_softmax(logits.detach().float(), dim=-1)
            self._prev_len = len(gen)
        # Mask everything except the allowed ids to -inf (in place, fp-safe).
        allowed = legal_token_ids(gen, contact_id=self.contact_id, end_id=self.end_id,
                                  remaining_full=self.remaining_full)
        mask = torch.ones_like(logits, dtype=torch.bool)
        idx = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
        mask[idx] = False
        logits[mask] = float("-inf")
        return logits

    def finalize(self, generated_ids: list[int]) -> None:
        """Record the last generated token's unmodified logprob (the processor is
        not called again after the final token). Idempotent-ish: only appends the
        single trailing token the lazy loop could not reach."""
        if (self.capture_nll and self._prev_logrow is not None
                and self._prev_len is not None
                and len(self.token_logprobs) == self._prev_len < len(generated_ids)):
            self.token_logprobs.append(float(self._prev_logrow[generated_ids[self._prev_len]]))

    def struct_nll(self) -> float:
        """Unmodified NLL of the generated structure section (contacts + ``<end>``)."""
        return -float(sum(self.token_logprobs))

    def max_new_tokens(self) -> int:
        """Exact generated length of a complete rollout: 3 per contact + ``<end>``."""
        return 3 * len(self.remaining_full) + 1
