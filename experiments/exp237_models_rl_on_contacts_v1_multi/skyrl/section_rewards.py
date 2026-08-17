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

import numpy as np

import consensus as cs
import contact_rewards as cr

#: Guard for the degenerate prompt where every section scores identically.
_STD_EPS = 1e-8

Pair = tuple[int, int]

REWARD_MODES = ("section_consensus", "final_f1", "best_f1")


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
    "scalar_reward",
    "scored_length",
    "section_bounds",
    "section_f1s",
    "token_advantages",
    "walk_rollout",
]
