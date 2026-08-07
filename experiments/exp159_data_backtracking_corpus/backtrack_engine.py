# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-in-the-loop synthesis of a contacts-v1 backtracking document (#159).

This is the **pure state machine** — no torch, no marinfold model, no GPU. It
manages the two-stream construction and the retraction policy; the base model
is reached only through two injected callables (a ``Proposer`` and a
``Scorer``), so the whole loop is unit-testable with a stub backend. The GPU
adapter that implements those callables from `contacts-v1-exp120-1.5B` lives
in the worker script (added for the pilot).

The design (see the experiment README and issue #159):

- **Two streams.** We build the *output document*'s structure section as an
  ordered edit list of ``("contact"|"retract", pair)`` statements, while the
  base model is always conditioned on a *clean* prompt containing only the
  currently-**live** contacts (no ``<retract>`` — it never trained on one).
  After each retraction we re-condition on the corrected live set. Here that
  is implicit: ``Proposer.propose(live)`` is handed the current live set, so
  every proposal is conditioned on the coherent post-retraction state.

- **Timing = posterior collapse.** A queued false positive is retracted when
  the base model's own belief in it — ``s(c)`` scored against the *committed*
  set (live minus the queue) — collapses relative to its peak, or drops below
  a floor, or falls out of the top-R of the predicted map. Timing therefore
  depends only on the visible contact set (learnable from context); ground
  truth is used only for the correctness guarantees below, never for timing.

- **Noise retraction.** With a small probability a *true* contact is
  **forcibly** retracted (the model is confident in it, so its posterior won't
  collapse — this must be forced, not queued) and re-emitted later, teaching
  that retraction is exploratory and reversible.

- **Correctness = a budget-reserved flush.** Whatever the trajectory, the
  main loop always leaves enough budget to (a) retract every still-live
  non-GT pair and (b) emit every missing GT pair, so the final live set
  equals GT exactly (recall philosophy F). Running out of context is the only
  way an FP survives — reported (``truncated``), not silently allowed.

The engine is agnostic to token rendering: it returns the ordered edit list
(and rich metrics); the caller prepends the sequence prefix and renders
tokens. ``read.live_contacts`` on the rendered document must equal GT.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

# Canonical (min, max) position pair — the same convention as read.py.
Pair = tuple[int, int]

CONTACT = "contact"
RETRACT = "retract"


def canon(a: int, b: int) -> Pair:
    return (a, b) if a <= b else (b, a)


class Proposer(Protocol):
    """The base model's next-contact proposal given the current live set.

    Returns a canonical position pair the model would emit next (conditioned
    on a clean prompt built from ``live``), or ``None`` to stop (EOS).
    """

    def propose(self, live: Sequence[Pair]) -> Pair | None: ...


class Scorer(Protocol):
    """The base model's per-pair belief ``s(c)`` given a committed context.

    ``score(committed, targets)`` returns ``{pair: s}`` for each target pair,
    where ``s`` is the model's probability of asserting that pair conditioned
    on ``committed`` (the live set with the queue removed) — one teacher-forced
    ``_fwd_matrix`` pass in the real adapter.
    """

    def score(
        self, committed: Sequence[Pair], targets: Sequence[Pair]
    ) -> dict[Pair, float]: ...


@dataclass(frozen=True)
class ProposeRequest:
    """Engine -> driver: "what contact comes next, given this live set?"."""

    live: list["Pair"]


@dataclass(frozen=True)
class ScoreRequest:
    """Engine -> driver: "score these queued pairs against this committed set"."""

    committed: list["Pair"]
    targets: list["Pair"]


@dataclass(frozen=True)
class RetractionPolicy:
    """Knobs for *when* queued contacts are retracted (calibrated on the pilot)."""

    # A queued/live pair is eligible for retraction only after this many
    # committed statements since it was emitted (never retract immediately).
    min_delay: int = 3
    # Re-score the queue every this many newly committed contacts (cost knob).
    eval_cadence: int = 2
    # Relative collapse: retract when s < tau * (max s seen while live).
    tau: float = 0.35
    # Absolute floor: retract when s < s_floor.
    s_floor: float = 1e-3
    # Rank drop: retract when the pair falls out of the top (rank_factor * |GT|)
    # entries of the scored set. None disables the rank trigger.
    rank_factor: float | None = None
    # Probability, per committed true contact, of a forced *noise* retraction.
    noise_retract_prob: float = 0.0
    # After this many retract cycles on the same pair, ban it (loop guard).
    loop_cap: int = 2


@dataclass
class _Queued:
    pair: Pair
    emitted_at: int          # committed-statement index at emission
    max_s: float = 0.0       # peak belief seen while live
    last_s: float = float("nan")


@dataclass
class BacktrackResult:
    """The synthesised edit list plus metrics for corpus QA (#159 criteria)."""

    statements: list[tuple[str, int, int]]
    live_final: frozenset[Pair]
    gt: frozenset[Pair]
    # Per-retraction: (pair, delay_in_statements, was_true_contact, trigger).
    retractions: list[tuple[Pair, int, bool, str]] = field(default_factory=list)
    n_contact_statements: int = 0
    n_retract_statements: int = 0
    n_reemit: int = 0
    truncated: bool = False   # ran out of budget before a clean flush

    @property
    def correct(self) -> bool:
        """Final live set equals GT exactly (the corpus's hard invariant)."""
        return self.live_final == self.gt

    def render_structure(self) -> str:
        """The structure-section token string (for folding / assembly)."""
        toks: list[str] = []
        for kind, a, b in self.statements:
            toks += [f"<{kind}>", f"<p{a}>", f"<p{b}>"]
        return " ".join(toks)


def backtracking_structure_gen(
    gt: frozenset[Pair],
    policy: RetractionPolicy,
    *,
    max_statements: int,
    rng: random.Random,
):
    """Generator form of the engine — yields model requests, returns a result.

    Yields :class:`ProposeRequest` / :class:`ScoreRequest` and expects the
    driver to ``send()`` back the proposed pair (or ``None``) / the score dict.
    Returns the :class:`BacktrackResult` as the generator's ``StopIteration``
    value. Decoupling the control flow from the model calls this way lets a
    driver run **many proteins concurrently** and batch their requests on the
    GPU (see ``batch_runner.py``); :func:`build_backtracking_structure` is the
    single-protein synchronous driver over this same logic.

    ``max_statements`` bounds the total number of statements (contacts +
    retracts + re-emits) so the caller's token budget is respected; the loop
    reserves enough of it for the closing flush. ``gt`` is the canonical
    ground-truth pair set the final live set must equal.
    """
    live: list[Pair] = []
    live_set: set[Pair] = set()
    emitted_at_step: dict[Pair, int] = {}
    queue: list[_Queued] = []            # false positives awaiting retraction
    pending_reemit: list[Pair] = []      # noise-retracted trues owed a re-emit
    retract_cycles: dict[Pair, int] = {}
    banned: set[Pair] = set()
    out: list[tuple[str, int, int]] = []

    res = BacktrackResult(statements=out, live_final=frozenset(), gt=gt)
    step = 0            # committed-contact counter (drives cadence + delays)

    def emit_contact(pair: Pair) -> None:
        nonlocal step
        out.append((CONTACT, pair[0], pair[1]))
        live.append(pair)
        live_set.add(pair)
        step += 1
        emitted_at_step[pair] = step
        res.n_contact_statements += 1

    def emit_retract(pair: Pair, *, delay: int, trigger: str) -> None:
        out.append((RETRACT, pair[0], pair[1]))
        live.remove(pair)
        live_set.discard(pair)
        res.n_retract_statements += 1
        res.retractions.append((pair, delay, pair in gt, trigger))
        retract_cycles[pair] = retract_cycles.get(pair, 0) + 1
        if retract_cycles[pair] >= policy.loop_cap:
            banned.add(pair)

    def committed() -> list[Pair]:
        q = {qc.pair for qc in queue}
        return [p for p in live if p not in q]

    def flush_needed() -> int:
        # Statements to reach live == gt: retract every live non-GT + emit
        # every missing GT (+1 margin for <end>, appended by the caller).
        return len(live_set - gt) + len(gt - live_set) + 1

    def budget_ok() -> bool:
        return len(out) + flush_needed() < max_statements

    since_eval = 0
    stopped = False
    no_progress = 0

    while budget_ok():
        # --- propose one contact (conditioned on the live set) -------------
        if not stopped:
            pair = yield ProposeRequest(list(live))
            if pair is None:
                stopped = True
            else:
                pair = canon(*pair)
                if pair in live_set or pair in banned:
                    no_progress += 1
                    if no_progress >= max_statements:
                        stopped = True
                else:
                    no_progress = 0
                    emit_contact(pair)
                    since_eval += 1
                    if pair not in gt:
                        queue.append(_Queued(pair, emitted_at=step))

        # --- forced noise retraction of a confident TRUE contact -----------
        if policy.noise_retract_prob > 0 and rng.random() < policy.noise_retract_prob:
            queued_pairs = {qc.pair for qc in queue}
            candidates = [
                p for p in live
                if p in gt
                and p not in queued_pairs
                and p not in pending_reemit
                and (step - emitted_at_step.get(p, step)) >= policy.min_delay
            ]
            if candidates and budget_ok():
                victim = rng.choice(candidates)
                emit_retract(
                    victim, delay=step - emitted_at_step[victim], trigger="noise"
                )
                pending_reemit.append(victim)

        # --- periodically re-score the queue and retract on collapse -------
        if queue and (since_eval >= policy.eval_cadence or stopped):
            since_eval = 0
            targets = [qc.pair for qc in queue]
            scores = yield ScoreRequest(committed(), targets)
            rank_cut = (
                None if policy.rank_factor is None
                else max(1, int(policy.rank_factor * max(len(gt), 1)))
            )
            rank_of: dict[Pair, int] = {}
            if rank_cut is not None:
                ranked = sorted(targets, key=lambda p: -scores.get(p, 0.0))
                rank_of = {p: i + 1 for i, p in enumerate(ranked)}
            still: list[_Queued] = []
            for qc in queue:
                s = scores.get(qc.pair, 0.0)
                qc.last_s = s
                qc.max_s = max(qc.max_s, s)
                delay = step - qc.emitted_at
                trigger = None
                if delay >= policy.min_delay and budget_ok():
                    if s < policy.s_floor:
                        trigger = "floor"
                    elif qc.max_s > 0 and s < policy.tau * qc.max_s:
                        trigger = "collapse"
                    elif rank_cut is not None and rank_of.get(qc.pair, 10**9) > rank_cut:
                        trigger = "rank"
                if trigger is not None:
                    emit_retract(qc.pair, delay=delay, trigger=trigger)
                else:
                    still.append(qc)
            queue = still

        # --- re-emit a pending (noise-retracted) true contact --------------
        if pending_reemit and (stopped or rng.random() < 0.5):
            p = pending_reemit[0]
            if p not in live_set and p not in banned and budget_ok():
                pending_reemit.pop(0)
                emit_contact(p)
                res.n_reemit += 1

        if stopped:
            break

    # --- closing flush: force live == gt exactly (philosophy F) -----------
    # Retract every still-live non-GT pair (queued FPs the trigger didn't
    # catch), then emit every missing GT pair (recall gap + any noise-retracted
    # true not yet re-emitted).
    for pair in list(live_set - gt):
        emit_retract(pair, delay=step - emitted_at_step.get(pair, step), trigger="flush")
    for pair in sorted(gt - live_set):
        if len(out) + 1 >= max_statements:
            res.truncated = True
            break
        was_pending = pair in pending_reemit
        emit_contact(pair)
        if was_pending:
            res.n_reemit += 1

    res.live_final = frozenset(live_set)
    return res


def build_backtracking_structure(
    gt: frozenset[Pair],
    proposer: Proposer,
    scorer: Scorer,
    policy: RetractionPolicy,
    *,
    max_statements: int,
    rng: random.Random,
) -> BacktrackResult:
    """Synchronous single-protein driver over :func:`backtracking_structure_gen`.

    Serves each yielded request immediately from the given ``proposer`` /
    ``scorer``. Behaviourally identical to the pre-generator engine, so the
    unit tests and the pilot keep using it unchanged.
    """
    gen = backtracking_structure_gen(
        gt, policy, max_statements=max_statements, rng=rng
    )
    try:
        request = next(gen)
        while True:
            if isinstance(request, ProposeRequest):
                response = proposer.propose(request.live)
            else:
                response = scorer.score(request.committed, request.targets)
            request = gen.send(response)
    except StopIteration as stop:
        return stop.value
