# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Section-level rewards for ``<contacts-v1.multi>`` — issue #237.

**The one idea this experiment tests.** #208 computed its reward on a *rollout*
and was scored on a *vote over 100 independent rollouts*. That is a unit
mismatch: the reward acted on an object the metric never sees, and the metric
scored an object no rollout could see. Under `<contacts-v1.multi>` a single
rollout emits ~22 candidate contact sets **inside one sequence**, so a reward
computed on the aggregate of one rollout's sections is computed on the same kind
of object the metric scores, and its credit assignment is *within* the sequence
where the policy gradient can reach it.

Everything here therefore operates on **sections of one rollout**, never on
rollouts of a group. `consensus.py` is imported unchanged — the leave-one-out
machinery is identical, only the population changes.

## The three rewards

| arm | reward | shape | estimator |
|---|---|---|---|
| **M-C** | ``m_k = C(all) − C(all \\ {k})``, the section's marginal contribution to its own rollout's consensus | per-section, dense | ``contacts_section`` |
| **M-F** | ``F1(last section)`` | one scalar per rollout | ``grpo`` |
| **M-B** | ``max_k F1(section k)`` — ORACLE | one scalar per rollout | ``grpo`` |
| **M-BC** | ``GRPO(max_k F1) + lam * GRPO(C_i(all))`` | two scalars per rollout, standardised separately | ``contacts_rollout`` |
| **M-FC** | ``GRPO(F1(last)) + lam * GRPO(C_i(all))`` — SYNTHESIS: write the consensus of your own drafts | two scalars per rollout | ``contacts_rollout`` |
| **M-K** | ``C_i(all)`` — the deployed metric itself, on the object the model emits | one scalar per rollout | ``grpo`` |

## The expectation calculation, done on paper first

#208 spent three training runs on modifications that broke ``E[r] = p − p̄`` by
weighting one side of a centred reward, each catchable beforehand by a few lines
of arithmetic. So, for M-C, explicitly:

Let ``m_k`` be section *k*'s marginal and let the group ``g`` be **every section
of every rollout sampled from this prompt** (``G`` rollouts x ~22 sections). The
advantage assigned to section *k* is

    A_k = (m_k − mean_g(m)) / (std_g(m) + eps)

so ``E_g[A] = 0`` **exactly**, by construction, for every prompt independently.
Consequences, each of which is the thing that went wrong somewhere in #208:

* *There is no first-order pressure on section count.* An extra section is worth
  emitting exactly when its marginal beats the group's mean marginal, and worth
  suppressing when it does not. A section that merely duplicates its siblings
  changes no vote and so scores ``m_k = 0``, which is **below** the mean and
  therefore negative — which is the pressure this model needs at #230's measured
  Jaccard of 0.304. A section that carries a true pair its siblings missed scores
  positive. Neither direction is free.
* *The normalisation is a division, not a re-weighting of one side.* ``std_g`` is
  computed over the same population as ``mean_g``, so it scales both signs
  identically and cannot tilt the zero point — which is exactly how
  ``err_decay``, the unweighted ``p̄`` and unnormalised novelty each broke in
  #208.
* *A prompt whose sections all score identically contributes zero.* ``std_g = 0``
  there, and the advantage is defined as 0 rather than as ``0/eps``. That is
  honest — the reward genuinely has nothing to say about that prompt — and
  `phase0_marginals.py` measures how often it happens before any GPU is booked.

## Assigning a per-section advantage to tokens

``A_k`` lands on **every response token of section k**, unscaled — *not* spread
as ``A_k / n_tokens``. This is deliberate and it is the difference between a
signal and a rounding error. SkyRL's GRPO gives one sequence-level scalar to
every token, so a per-token advantage of magnitude ~1 is the scale the optimiser
and the learning rate are calibrated for. Spreading ``A_k`` over a section's ~300
tokens instead would make M-C's gradient ~300x smaller than M-F's at the same
learning rate, and #208 already paid a full run for exactly this mistake in the
other direction (``lam_doc`` 4.5 carried 0.42 % of the stepwise term's spread —
"not a weak signal, no signal").

A section owns the ``<begin_statements>`` token that **opens** it, so the
decision to start another candidate is shaped by whether that candidate turned
out to be worth starting. The final section owns the ``<end>`` token.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import consensus as cs
import contact_rewards as cr

#: Guard for the degenerate prompt where every section scores identically.
_STD_EPS = 1e-8

Pair = tuple[int, int]

REWARD_MODES = ("section_consensus", "final_f1", "best_f1", "best_plus_consensus",
                "consensus_shaped",
                "final_plus_consensus", "consensus_only")


@dataclass
class RolloutSections:
    """One multi rollout, decoded into its candidate contact sets."""

    #: Dedup'd (min, max) sequence-index pairs, one set per section, in order.
    sections: list[set[Pair]]
    #: ``(start, end)`` response-token indices for each section, half-open.
    bounds: list[tuple[int, int]]
    n_response_tokens: int
    n_scored: int
    n_correct: int
    finished: bool
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def n_sections(self) -> int:
        return len(self.sections)


def section_bounds(response_ids: Sequence[int],
                   vocab: cr.ContactVocab = cr.DEFAULT_VOCAB) -> list[tuple[int, int]]:
    """Half-open response-token spans, one per section.

    The prompt ends **on** a ``<begin_statements>``, so response index 0 is
    already inside section 0 and every later ``<begin_statements>`` opens a new
    one. The marker is assigned to the section it opens (see the module
    docstring), and the tail — including ``<end>`` — belongs to the last section.

    Mirrors :func:`contact_rewards.walk_contacts`'s section numbering exactly:
    span *k* is the region whose contacts that function tags ``section == k``.
    """
    starts = [0]
    for i, tid in enumerate(response_ids):
        if tid == vocab.begin_id:
            starts.append(i)
    ends = starts[1:] + [len(response_ids)]
    return list(zip(starts, ends))


def scored_length(response_ids: Sequence[int],
                  vocab: cr.ContactVocab = cr.DEFAULT_VOCAB) -> int:
    """Response length up to and including the first ``<end>``.

    A multi document is closed by ``<end>``, and `contact_rewards.walk_contacts`
    stops there. The sampler does not: the natural stop is ``<eos>``, which the
    model emits *after* ``<end>``, and nothing prevents a rollout from running on
    into a second document. Bounding the spans here keeps a stray continuation
    from being folded into the last section's advantage — it simply receives
    zero, which is the honest credit for tokens the metric never reads.
    """
    for i, tid in enumerate(response_ids):
        if tid == vocab.end_id:
            return i + 1
    return len(response_ids)


def walk_rollout(
    response_ids: Sequence[int],
    pos_to_seq: Mapping[int, int],
    gt: set[Pair],
    *,
    vocab: cr.ContactVocab = cr.DEFAULT_VOCAB,
) -> RolloutSections:
    """Decode one multi rollout into per-section contact sets and token spans."""
    eff = scored_length(response_ids, vocab=vocab)
    contacts = cr.walk_contacts(response_ids, pos_to_seq, gt, starts_in_section=True, vocab=vocab)
    bounds = section_bounds(response_ids[:eff], vocab=vocab)
    sections: list[set[Pair]] = [set() for _ in bounds]
    n_scored = n_correct = 0
    for c in contacts:
        if c.reason == "truncated":
            continue
        n_scored += 1
        n_correct += int(c.correct)
        if c.pair is not None and c.reason == "ok" and c.section < len(sections):
            sections[c.section].add(c.pair)
    finished = eff < len(response_ids) or (response_ids and response_ids[-1] == vocab.end_id)
    union: set[Pair] = set().union(*sections) if sections else set()
    total_votes = sum(len(s) for s in sections)
    js = [
        len(a & b) / max(len(a | b), 1)
        for i, a in enumerate(sections) for b in sections[i + 1:]
        if (a or b)
    ]
    diagnostics = {
        "n_sections": float(len(sections)),
        "union_pairs": float(len(union)),
        "total_votes": float(total_votes),
        "votes_per_pair": (total_votes / len(union)) if union else 0.0,
        "mean_jaccard": float(np.mean(js)) if js else math.nan,
        "n_empty_sections": float(sum(1 for s in sections if not s)),
    }
    return RolloutSections(
        sections=sections, bounds=bounds, n_response_tokens=len(response_ids),
        n_scored=n_scored, n_correct=n_correct, finished=finished, diagnostics=diagnostics,
    )


def consensus_and_marginals(
    sections: Sequence[set[Pair]], gt: set[Pair], length: int
) -> tuple[float, np.ndarray]:
    """Within-rollout consensus R-precision and each section's LOO marginal.

    The candidate universe is every ``sep >= 6`` pair over ``length`` residues,
    ranked by vote count with the deployed metric's stable positional tie-break —
    `consensus.py` verbatim, only counting *sections* where #208 counted
    *rollouts*.

    Returns ``(nan, zeros)`` when the protein has no true contacts in band, which
    makes the caller's arithmetic total rather than conditional.
    """
    n_sections = len(sections)
    if n_sections == 0 or not gt or length <= 0:
        return math.nan, np.zeros(n_sections, dtype=np.float64)
    pairs, position = cs.candidate_index(length)
    is_true = cs.truth_mask(pairs, gt)
    n_true = int(is_true.sum())
    if n_true <= 0 or len(pairs) == 0:
        return math.nan, np.zeros(n_sections, dtype=np.float64)
    votes = cs.vote_counts(sections, position, len(pairs))
    consensus, marginals = cs.loo_marginals(votes, is_true, n_true)
    if math.isnan(consensus):
        return math.nan, np.zeros(n_sections, dtype=np.float64)
    return float(consensus), np.nan_to_num(marginals, nan=0.0)


def prefix_marginals(
    sections: Sequence[set[Pair]], gt: set[Pair], length: int
) -> np.ndarray:
    """Each section's **causal** contribution: ``C(s_1..s_k) - C(s_1..s_{k-1})``.

    The marginal against exactly what was in context when the section was
    written, rather than the leave-one-out marginal against all its siblings
    including ones that did not exist yet. This is the quantity that answers
    "given what you have already emitted, what did this add?" — so duplicating an
    earlier candidate earns nothing while covering a pair the predecessors missed
    earns a lot.

    **This is NOT safe as a standalone reward and was measured to be so.** It
    telescopes (``sum_k m_k = C(all) - C(empty)``), which looks like it bounds the
    total, but ``loss_reduction=token_mean`` reads the **mean**, not the sum — and
    a short rollout's early sections are scored against a near-empty prefix, so
    its mean is large. Measured group-centred advantage at one section: **+2.03**,
    against −0.22 at 22. It is the same count-adverse pathology that destroyed arm
    M-C, only milder.

    It is safe **only** where the caller makes it zero-sum within the rollout, on
    top of a scale-correct rollout-level base — which is how
    :func:`shaped_section_advantages` uses it.

    Returns:
        ``[K]`` marginals, all zero when the protein has no scoreable ground
        truth (so the caller's arithmetic stays total).
    """
    n = len(sections)
    if n == 0 or not gt or length <= 0:
        return np.zeros(n, dtype=np.float64)
    pairs, position = cs.candidate_index(length)
    is_true = cs.truth_mask(pairs, gt)
    n_true = int(is_true.sum())
    if n_true <= 0 or len(pairs) == 0:
        return np.zeros(n, dtype=np.float64)
    per_section = cs.vote_counts(sections, position, len(pairs))
    out = np.zeros(n, dtype=np.float64)
    running = np.zeros(len(pairs), dtype=np.int64)
    prev = 0.0                     # C of the empty prefix: nothing ranked, 0.0
    for k in range(n):
        running += per_section[k]
        cur = cs.rprecision(running, is_true, n_true)
        cur = 0.0 if math.isnan(cur) else float(cur)
        out[k] = cur - prev
        prev = cur
    return out


def novelty_marginals(
    sections: Sequence[set[Pair]], gt: set[Pair], length: int
) -> np.ndarray:
    """What section *k* added that its predecessors had not: ``(new_true - new_false) / R``.

    Arm M-KS3. The direct form of the thing arm M-KS2 was reaching for through a
    proxy. Measured on 11,516 real sections, M-KS2's causal prefix marginal
    correlates only **+0.194** with actual novelty — it is tilted the right way
    and mostly noise — yet it still produced the best ORACLE candidates in #237
    (0.5677). This asks what the signal it was approximating is worth on its own.

    For section ``k`` with pair set ``s_k`` and prefix union ``U = s_1 ∪ … ∪
    s_{k-1}``, over pairs inside the ``sep >= 6`` band::

        new    = s_k \ U
        m_k    = ( |new ∩ gt| - |new \ gt| ) / |gt|

    **Why the false term is there and not optional.** ``|new ∩ gt| / R`` alone —
    plain recall gain — pays for *volume*: a section that dumps a hundred junk
    pairs catches new true ones by chance and scores positive for it. #237 has
    already watched that failure once, when arm M-F ran to 259 sections carrying
    1.4 contacts each. Subtracting the new *false* pairs makes the term "did the
    union get better or worse", so padding is priced rather than rewarded.

    **Normalised by R, not by section size.** R-precision cuts a ranking at
    ``R = |gt|``, so a pair is worth what it is worth relative to the budget the
    metric actually spends — dividing by ``|s_k|`` instead would make a tiny
    accurate section look identical to a large one, and the aggregate cares about
    the count.

    Returns:
        ``[K]`` marginals, all zero when the protein has no scoreable ground
        truth. The first section is scored against an empty prefix and so scores
        its whole precision-adjusted content — the same positional artefact that
        killed the un-corrected arm M-KS, and the reason this must be used with
        :func:`positional_baseline`.
    """
    n = len(sections)
    if n == 0 or not gt or length <= 0:
        return np.zeros(n, dtype=np.float64)
    band = {p for p in gt}
    r = float(len(band))
    if r <= 0:
        return np.zeros(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    seen: set = set()
    for k, sec in enumerate(sections):
        new = sec - seen
        if new:
            hits = len(new & band)
            out[k] = (hits - (len(new) - hits)) / r
        seen |= sec
    return out


def pair_token_advantages(
    response_ids: Sequence[int],
    pos_to_seq: Mapping[int, int],
    gt: set[Pair],
    *,
    lam_false: float = 1.0,
    vocab: cr.ContactVocab = cr.DEFAULT_VOCAB,
) -> np.ndarray:
    """Per-TOKEN shaping scored per emitted pair — arm M-KP.

    The section stops being the credit unit. Each ``<contact> <pI> <pJ>`` triple
    is scored on its own and its value lands on its own three tokens::

        first time this pair appears in the rollout, and it is TRUE   ->  +1 / R
        first time this pair appears in the rollout, and it is FALSE  ->  -lam / R
        the pair has already appeared anywhere earlier                ->   0

    **Why this is partition-invariant, where the section forms were not.** Arms
    M-KS and M-KS3 both failed because a section's value depended on how the
    predictions were sliced — on the section's index (decaying, so "stop early")
    and on its size (finer slicing scores better, so "fragment"). Here the
    partition never enters the arithmetic: cutting a section in two leaves every
    pair's value and every pair's tokens exactly as they were.

    **The novelty gate is what keeps this from being a precision reward.** #237
    excludes per-contact-only rewards because #208 established they are
    sharpening operators. This is not that, in two respects: it is *shaping* on
    top of arm M-K's scale-correct rollout-level base rather than the whole
    objective, and a pair already emitted scores **exactly zero whether it is
    true or false**. A policy that sharpens by repeating its confident set earns
    nothing after the first section; the only way to score is to add correct
    content that is not already there.

    **Structural tokens carry no shaping, and that is the load-bearing choice.**
    ``<begin_statements>``, ``<end>`` and everything outside a triple keep a
    shaping value of exactly 0, and the centring below runs over the triple
    tokens *only*. Centring over all tokens instead would give every structural
    token ``-mean``, and since most emitted pairs are false the mean is negative
    — so each ``<begin_statements>`` would receive a **positive** advantage for
    existing. That is precisely arm M-KS3's runaway, reachable here without a
    single new pair being emitted. Under this construction the decision to open
    a section receives no shaping signal in either direction.

    Returns:
        ``[len(response_ids)]`` shaping values, zero everywhere except the
        contact triples, and summing to exactly zero across them.
    """
    n = len(response_ids)
    out = np.zeros(n, dtype=np.float64)
    if not gt or n == 0:
        return out
    r = float(len(gt))
    eff = scored_length(response_ids, vocab=vocab)
    contacts = cr.walk_contacts(response_ids, pos_to_seq, gt, starts_in_section=True, vocab=vocab)
    seen: set[Pair] = set()
    spans: list[tuple[int, int]] = []
    for c in contacts:
        if c.reason != "ok" or c.pair is None or c.start >= eff:
            continue
        lo, hi = c.start, min(c.start + 3, eff)     # <contact> <pI> <pJ>
        if hi <= lo:
            continue
        if c.pair in seen:
            val = 0.0
        else:
            val = (1.0 if c.pair in gt else -float(lam_false)) / r
            seen.add(c.pair)
        out[lo:hi] = val
        spans.append((lo, hi))
    if not spans:
        return out
    mask = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        mask[lo:hi] = True
    out[mask] -= out[mask].mean()                   # zero-sum over the pair tokens ONLY
    return out


def positional_baseline(marginals_by_rep: Mapping) -> np.ndarray:
    """Mean prefix marginal at each section POSITION, across a prompt group.

    The correction arm M-KS needed and did not have. ``prefix_marginals`` decays
    in ``k`` **by construction** — ``C(s_1..s_k)`` saturates, and the first
    section is scored against an empty prefix, so it captures nearly the whole
    telescoped total. Measured on 566 real rollouts: the centred term is
    **+0.357** at ``k = 0`` and negative at every ``k >= 2``, with a negative
    slope in **100 %** of rollouts.

    Centring within the rollout does not remove that. Zero-sum bounds the
    rollout's total advantage — its *level* — while leaving the *shape* intact,
    and because a section owns the ``<begin_statements>`` token that **opens**
    it, a term decreasing in ``k`` is a direct penalty on the decision to write
    another candidate. Arm M-KS collapsed to 10.66 sections by step 21, the
    fastest count collapse in #237, entirely inside its own zero-sum guarantee.

    Subtracting the group's mean marginal *at the same position* removes the
    deterministic trend and leaves the question that was wanted all along: did
    this section do better than a typical ``k``-th section?

    Args:
        marginals_by_rep: ``{repetition_id: marginals}`` for ONE prompt group.

    Returns:
        ``[max_K]`` positional means. Position ``k`` averages only over the
        rollouts that actually reached ``k``, so a group of ragged lengths is
        handled without padding a zero into the mean.
    """
    arrays = [np.asarray(m, dtype=np.float64) for m in marginals_by_rep.values()]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return np.zeros(0, dtype=np.float64)
    width = max(a.size for a in arrays)
    out = np.zeros(width, dtype=np.float64)
    for k in range(width):
        vals = [a[k] for a in arrays if a.size > k]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def shaped_section_advantages(
    base: float, marginals: np.ndarray, beta: float,
    positional: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``base + beta * (m_k - mean_k m)`` — arm M-KS's per-token advantage source.

    Args:
        base: The rollout's already-standardised scalar advantage, i.e.
            ``GRPO_group(C_i(all))_i``. Arm M-K is this alone.
        marginals: ``[K]`` per-section marginals, from :func:`prefix_marginals`.
        beta: Shaping weight. ``0.0`` reduces this to arm M-K exactly.

    Returns:
        ``[K]`` per-section advantages.

    **The shaping term is centred within the rollout, and that is the whole
    safety argument.** ``sum_k (m_k - mean_k m) = 0`` by construction, so no
    matter how the marginals scale with the candidate count, the term contributes
    **nothing** to the rollout's total advantage — it can only redistribute the
    base among the sections that earned it. Every count pathology in this
    experiment (M-C's +4.79 at one section, the prefix form's +2.03) was a
    statement about a reward's *level* as a function of ``K``; a zero-sum term has
    no level to move.

    The base carries the level, and it is scale-correct on its own: ``C_i(all)``
    *falls* when sections are dropped (0.543 at 22, 0.341 at one).
    """
    if len(marginals) == 0:
        return np.zeros(0, dtype=np.float64)
    adv = np.full(len(marginals), float(base), dtype=np.float64)
    if beta != 0.0:
        m = np.asarray(marginals, dtype=np.float64)
        if positional is not None and len(positional):
            # Remove the deterministic decay in k FIRST, then centre. Centring
            # alone leaves the positional trend intact and turns the term into a
            # "stop early" signal -- see `positional_baseline`.
            b = np.asarray(positional, dtype=np.float64)
            m = m - (b[:len(m)] if len(b) >= len(m)
                     else np.concatenate([b, np.zeros(len(m) - len(b))]))
        adv = adv + float(beta) * (m - m.mean())
    return adv


def section_f1s(sections: Sequence[set[Pair]], gt: set[Pair]) -> list[float]:
    """``all``-band F1 of each section, via `contact_rewards.f1_all_band`.

    NaN (empty ground truth) is mapped to 0.0 so ``max`` and ``[-1]`` are total.
    """
    out = []
    for s in sections:
        v = cr.f1_all_band(s, gt)
        out.append(0.0 if math.isnan(v) else float(v))
    return out


def scalar_reward(mode: str, walk: RolloutSections, gt: set[Pair]) -> float:
    """The one-scalar-per-rollout rewards: arm M-F and arm M-B.

    Args:
        mode: ``"final_f1"`` (the last section the rollout commits to) or
            ``"best_f1"`` (the best section — **ORACLE**, it selects with ground
            truth and is reported as a ceiling).
        walk: The decoded rollout.
        gt: In-band ground-truth pairs.

    Returns:
        The reward. A rollout with no sections scores 0.0, which is the correct
        floor: under a GRPO group baseline that is a negative advantage against
        siblings that produced something.
    """
    f1s = section_f1s(walk.sections, gt)
    if not f1s:
        return 0.0
    if mode == "final_f1":
        return f1s[-1]
    if mode == "best_f1":
        return max(f1s)
    raise ValueError(f"scalar_reward does not serve mode {mode!r}")


def count_penalty(n_sections: int, beta: float, floor: float) -> float:
    """``beta * min(0, K - floor)`` — a one-sided barrier on the candidate count.

    Added to the RAW scalar reward, before :func:`grpo_standardise`. Three
    properties, each of which is the reason for a choice made elsewhere:

    **The deadband is exact.** For ``K >= floor`` this is identically ``0.0``,
    and because GRPO subtracts the group mean, a term that is the same constant
    for every rollout in a group changes the standardised advantage by exactly
    nothing. So a healthy batch is untouched — not "barely touched". That is why
    the term is added raw rather than standardised on its own: ``GRPO`` of a
    near-constant column amplifies noise into a ±1 signal, which would put
    section-count pressure on batches that have no section-count problem.

    **It is bounded.** At ``K = 1`` with the defaults it is ``-0.51`` against a
    reward in ``[0, 1]`` — decisive but finite. Arm M-C's per-section marginal
    reached ``+4.79`` at one section and **366x** its value at 22, which is why
    no fixed weight could balance it inside a blend. This cannot run away.

    **It cannot pay for padding.** ``min(0, .)`` means there is no gradient
    above the floor, so the policy is never paid for emitting a 60th section.
    The failure mode a two-sided count bonus would create — arm M-F's, 259
    sections carrying 1.4 contacts each — is unreachable from here. Watch
    ``multi/empty_sections`` anyway.

    Args:
        n_sections: The rollout's candidate count ``K``.
        beta: Scale. ``0.0`` disables the term, which is the default everywhere.
        floor: The count below which the penalty engages.

    Returns:
        ``0.0`` when ``n_sections >= floor``, else ``beta * (n_sections - floor)``.
    """
    if beta == 0.0:
        return 0.0
    return float(beta * min(0.0, float(n_sections) - float(floor)))


def grpo_standardise(values: Sequence[float]) -> np.ndarray:
    """SkyRL's GRPO group normalisation, reproduced exactly.

    This is the ``GRPO(·)`` written in the README and RESULTS, and it is
    reproduced here rather than described because two of its details are easy to
    get wrong and neither is visible from the name.

    From `skyrl/backends/skyrl_train/utils/ppo_utils.py::compute_grpo_outcome_advantage`,
    for a group ``g`` of rollouts sharing a prompt, with per-rollout scalar
    reward ``R_i``::

        A_i = (R_i - mean_g(R)) / (std_g(R) + 1e-6)

    * ``std`` is `torch.std`, the **unbiased sample** standard deviation
      (``ddof = 1``) — *not* numpy's default population sd (``ddof = 0``). On a
      group of 8 that is a 7 % difference in the denominator.
    * ``epsilon = 1e-6`` is added to the **standard deviation**, not to the
      variance.
    * A **singleton** group takes ``mean = 0`` and ``std = 1``, i.e. the raw
      reward passes through uncentred. That is SkyRL's behaviour, not an
      approximation of it.

    The resulting scalar is then assigned to **every response token** of rollout
    ``i`` (``scores.unsqueeze(-1) * response_mask``), so with
    ``loss_reduction=token_mean`` a rollout's contribution to the loss is
    proportional to its own token count — longer rollouts carry more gradient.

    Args:
        values: One scalar reward per rollout, for a SINGLE prompt group.

    Returns:
        The standardised advantages, one per rollout, in the same order.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return arr.copy()                       # SkyRL: mean 0, std 1
    return (arr - arr.mean()) / (arr.std(ddof=1) + 1e-6)


def centred_section_advantages(marginals_by_key: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Centre and scale every section marginal against the whole prompt group.

    Args:
        marginals_by_key: One rollout's marginal vector per rollout key, for a
            SINGLE prompt. Pooling across prompts would let an easy protein's
            marginal scale swamp a hard one's — the same reason GRPO normalises
            per group rather than per batch.

    Returns:
        The same mapping with ``(m - mean) / (std + eps)`` applied. When the
        pooled spread is zero the advantages are exactly zero: the reward has
        nothing to say about this prompt, and saying so is better than dividing
        by ``eps`` and amplifying float noise into a gradient.
    """
    pooled = np.concatenate([v for v in marginals_by_key.values() if v.size]) \
        if any(v.size for v in marginals_by_key.values()) else np.zeros(0)
    if pooled.size == 0:
        return {k: np.zeros_like(v) for k, v in marginals_by_key.items()}
    mean = float(pooled.mean())
    std = float(pooled.std())
    if std <= _STD_EPS:
        return {k: np.zeros_like(v) for k, v in marginals_by_key.items()}
    return {k: (v - mean) / std for k, v in marginals_by_key.items()}


def token_advantages(advantages: np.ndarray, bounds: Sequence[tuple[int, int]],
                     n_response_tokens: int) -> np.ndarray:
    """Broadcast a per-section advantage onto that section's response tokens.

    Unscaled, not divided by the section length — see the module docstring. Any
    token outside every section span (there are none in practice; the spans
    tile the response) stays at 0.
    """
    out = np.zeros(int(n_response_tokens), dtype=np.float32)
    for a, (start, end) in zip(advantages, bounds):
        if end > start:
            out[start:end] = np.float32(a)
    return out


__all__ = [
    "REWARD_MODES",
    "RolloutSections",
    "centred_section_advantages",
    "consensus_and_marginals",
    "count_penalty",
    "novelty_marginals",
    "pair_token_advantages",
    "positional_baseline",
    "prefix_marginals",
    "shaped_section_advantages",
    "grpo_standardise",
    "scalar_reward",
    "scored_length",
    "section_bounds",
    "section_f1s",
    "token_advantages",
    "walk_rollout",
]
