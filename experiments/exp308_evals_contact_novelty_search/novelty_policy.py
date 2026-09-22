"""Sample complete beam contacts after penalizing prior-rollout frequency."""

import math
import random
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class PairCandidate:
    """A valid two-token continuation and its model joint log probability."""

    token_ids: tuple[int, int]
    pair: tuple[int, int]
    logprob: float
    rank: int


def effective_epsilon(initial: float, contact_index: int, decay_contacts: int) -> float:
    """Linearly retire novelty pressure after early contact emissions.

    A zero decay length keeps a constant penalty for the ablation.
    """
    if initial < 0 or not math.isfinite(initial):
        raise ValueError("initial epsilon must be finite and nonnegative")
    if contact_index < 0 or decay_contacts < 0:
        raise ValueError("contact index and decay length must be nonnegative")
    if decay_contacts == 0:
        return initial
    return initial * max(0.0, 1.0 - contact_index / decay_contacts)


def pair_candidates(
    sequences: Sequence,
    prompt_length: int,
    position_to_index: dict[int, int],
    minimum_separation: int = 6,
) -> list[PairCandidate]:
    """Keep beam continuations that form a contact on the input sequence."""

    candidates = []
    for rank, sequence in enumerate(sequences):
        tokens = tuple(int(token) for token in sequence.tokens[prompt_length:])
        if len(tokens) != 2 or tokens[0] not in position_to_index or tokens[1] not in position_to_index:
            continue
        a, b = position_to_index[tokens[0]], position_to_index[tokens[1]]
        if abs(a - b) < minimum_separation:
            continue
        logprob = float(sequence.cum_logprob)
        if not math.isfinite(logprob):
            raise ValueError("non-finite beam log probability")
        candidates.append(PairCandidate(tokens, (min(a, b), max(a, b)), logprob, rank))
    return candidates


def sample_candidate(
    candidates: Sequence[PairCandidate],
    rng: random.Random,
    previous_counts: Mapping[tuple[int, int], int],
    epsilon: float,
) -> tuple[PairCandidate, float, int]:
    """Softmax joint log probabilities minus epsilon times previous usage.

    The count is the number of previously completed rollouts containing the
    contact. All candidates in one wave see the same completed-wave counts.
    """

    if not candidates:
        raise ValueError("cannot sample from an empty pair beam")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and nonnegative")
    adjusted = [candidate.logprob - epsilon * previous_counts.get(candidate.pair, 0)
                for candidate in candidates]
    best = max(adjusted)
    weights = [math.exp(score - best) for score in adjusted]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for candidate, score, weight in zip(candidates, adjusted, weights):
        cumulative += weight
        if threshold <= cumulative:
            return candidate, score, previous_counts.get(candidate.pair, 0)
    final = candidates[-1]
    return final, adjusted[-1], previous_counts.get(final.pair, 0)
