"""Choose a complete contact pair from short, reference-free beam expansions."""

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairCandidate:
    """A valid two-token continuation and its model joint log probability."""

    token_ids: tuple[int, int]
    pair: tuple[int, int]
    logprob: float
    rank: int


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


def sample_candidate(candidates: Sequence[PairCandidate], rng: random.Random) -> PairCandidate:
    """Sample a complete pair by its joint model probability within the beam."""

    if not candidates:
        raise ValueError("cannot sample from an empty pair beam")
    best = max(candidate.logprob for candidate in candidates)
    weights = [math.exp(candidate.logprob - best) for candidate in candidates]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if threshold <= cumulative:
            return candidate
    return candidates[-1]
