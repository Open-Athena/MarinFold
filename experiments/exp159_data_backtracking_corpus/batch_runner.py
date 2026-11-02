# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive many proteins' backtracking engines concurrently, batching on the GPU.

The single-protein driver runs the model at **batch size 1**: each `propose`
is one short autoregressive generation, so the GPU sits mostly idle and a
document costs ~13 s. Profiling attributed ~86% of that to `propose` (~50
calls x ~220 ms) and only ~14% to `score`.

This module fixes both halves of that:

- **Cross-protein batching.** Each protein's engine is a generator
  (``backtracking_structure_gen``) that *yields* its model requests, so N of
  them can be advanced in lockstep: every scheduler tick collects all pending
  ``ProposeRequest``s and serves them in **one** padded batch. With N proteins
  in flight the per-document cost amortises down by roughly the batch factor.
- **A shorter decode budget.** A contact statement is exactly 3 tokens
  (``<contact> <pX> <pY>``), so generating 12 was ~4x more decode steps than
  needed. We generate a handful and take the first statement. A *duplicate*
  proposal is returned as-is rather than treated as EOS — the engine already
  skips live pairs (and counts no-progress), so this cannot truncate a
  document early.

Left padding is used for the batch (decoder-only generation), so short prompts
never see their own pad tokens as context.

``score`` requests are served per-protein (they are already internally batched
over tails by the backend, and are the minority of the cost).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import torch

from backtrack_engine import (
    BacktrackResult,
    ProposeRequest,
    RetractionPolicy,
    backtracking_structure_gen,
    canon,
)

from marinfold.document_structures.contacts_v1.read import iter_structure_statements

# Tokens generated per propose. A contact statement is 3; a little slack
# absorbs a stray leading token without paying for 12 decode steps.
DEFAULT_PROPOSE_TOKENS = 6


def _left_padded_batch(prompts: Sequence[Sequence[int]], pad_id: int, device):
    """Left-pad ragged prompts into ``(input_ids, attention_mask)`` tensors."""
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        attention[row, width - len(prompt):] = 1
    return input_ids.to(device), attention.to(device), width


def batched_propose(
    backend,
    adapters: Sequence,
    lives: Sequence[Sequence[tuple[int, int]]],
    *,
    max_new_tokens: int = DEFAULT_PROPOSE_TOKENS,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    chunk: int = 32,
) -> list[tuple[int, int] | None]:
    """One proposed pair per (adapter, live) — served in padded GPU batches.

    Returns the canonical seq-index pair each protein's model would emit next,
    or ``None`` where it emitted ``<end>`` (or nothing parseable).
    """
    tokenizer = backend.tokenizer
    model = backend._model          # experiment script: use the backend's model
    device = backend._device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<pad>")

    out: list[tuple[int, int] | None] = [None] * len(adapters)
    for start in range(0, len(adapters), chunk):
        sl = slice(start, start + chunk)
        group = list(zip(adapters[sl], lives[sl], strict=True))
        prompts = [ad._prompt_ids(live) for ad, live in group]
        input_ids, attention, width = _left_padded_batch(prompts, pad_id, device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                attention_mask=attention,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=adapters[start].end_id,
            )
        for offset, (adapter, _live) in enumerate(group):
            completion = generated[offset, width:].tolist()
            text = tokenizer.decode(completion, skip_special_tokens=False)
            out[start + offset] = _first_contact_pair(adapter, text)
    return out


def _first_contact_pair(adapter, text: str) -> tuple[int, int] | None:
    """First ``<contact>`` statement in ``text`` as a seq-index pair, else None.

    A pair already live is still returned: the engine skips live pairs (and
    counts no-progress), so surfacing it is correct and avoids mistaking a
    duplicate for EOS.
    """
    for kind, a, b in iter_structure_statements(text):
        if kind != "contact":
            continue
        ia, ib = adapter.pos_to_seq.get(a), adapter.pos_to_seq.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        return canon(ia, ib)
    return None


def run_batched(
    jobs: Sequence[tuple[str, frozenset, object, int]],
    backend,
    policy: RetractionPolicy,
    *,
    seed: int = 0,
    propose_tokens: int = DEFAULT_PROPOSE_TOKENS,
    chunk: int = 32,
) -> dict[str, BacktrackResult]:
    """Run every job's engine concurrently, batching propose across proteins.

    ``jobs`` is ``(entry_id, gt, adapter, max_statements)`` per protein. Returns
    ``{entry_id: BacktrackResult}``. Every tick: advance all engines to their
    next request, serve all pending proposes in one batch, serve any score
    requests per-protein, repeat until all engines finish.
    """
    state: dict[str, dict] = {}
    for entry_id, gt, adapter, max_statements in jobs:
        gen = backtracking_structure_gen(
            gt, policy, max_statements=max_statements,
            # Deterministic per-protein seed: str.__hash__ is randomized per
            # process, so derive it from sha1 instead.
            rng=random.Random(
                int(hashlib.sha1(f"{seed}:{entry_id}".encode()).hexdigest()[:8], 16)
            ),
        )
        try:
            state[entry_id] = {"gen": gen, "adapter": adapter, "req": next(gen)}
        except StopIteration as stop:
            state[entry_id] = {"gen": None, "adapter": adapter, "result": stop.value}

    results: dict[str, BacktrackResult] = {}
    while True:
        active = [k for k, v in state.items() if v.get("gen") is not None]
        if not active:
            break

        # 1. All pending proposes -> one batched GPU call.
        propose_keys = [k for k in active if isinstance(state[k]["req"], ProposeRequest)]
        responses: dict[str, object] = {}
        if propose_keys:
            pairs = batched_propose(
                backend,
                [state[k]["adapter"] for k in propose_keys],
                [state[k]["req"].live for k in propose_keys],
                max_new_tokens=propose_tokens,
                chunk=chunk,
            )
            responses.update(dict(zip(propose_keys, pairs, strict=True)))

        # 2. Score requests, per protein (already tail-batched internally).
        for key in active:
            if key in responses:
                continue
            req = state[key]["req"]
            responses[key] = state[key]["adapter"].score(req.committed, req.targets)

        # 3. Advance every engine with its response.
        for key in active:
            entry = state[key]
            try:
                entry["req"] = entry["gen"].send(responses[key])
            except StopIteration as stop:
                entry["gen"] = None
                results[key] = stop.value

    for key, entry in state.items():
        if key not in results and "result" in entry:
            results[key] = entry["result"]
    return results
