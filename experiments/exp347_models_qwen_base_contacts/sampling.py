"""Finish capped samples without discarding or resampling their generated prefixes."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """One generated suffix and its backend termination reason."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str


Generate = Callable[[list[list[int]], list[int], list[int]], list[Sample]]


def finish_rollouts(
    samples: list[Sample],
    prompt_ids: list[list[int]],
    initial_budget: int,
    context_length: int,
    seeds: list[int],
    generate: Generate,
) -> tuple[list[Sample], list[int]]:
    """Extend only length-capped samples, retaining all earlier sampled tokens.

    Independent continuation RNG streams sample the same conditional next-token
    distribution. Already terminated rollouts are never resampled. Continuations
    grow geometrically up to the fixed model context; a still-capped sample is
    returned as incomplete so the caller refuses to publish a metric.
    """
    if not len(samples) == len(prompt_ids) == len(seeds):
        raise ValueError("Sample, prompt, and seed counts differ")
    result = list(samples)
    rounds = [0] * len(samples)
    while True:
        pending = [
            i
            for i, s in enumerate(result)
            if s.finish_reason == "length"
            and len(prompt_ids[i]) + len(s.token_ids) < context_length
        ]
        if not pending:
            return result, rounds
        limits = [
            min(
                initial_budget * 2 ** rounds[i],
                context_length - len(prompt_ids[i]) - len(result[i].token_ids),
            )
            for i in pending
        ]
        prefixes = [prompt_ids[i] + list(result[i].token_ids) for i in pending]
        continuation_seeds = [
            (seeds[i] + (rounds[i] + 1) * 1_000_003) % (2**31) for i in pending
        ]
        generated = generate(prefixes, limits, continuation_seeds)
        if len(generated) != len(pending):
            raise ValueError("Continuation count differs from pending rollouts")
        for i, limit, chunk in zip(pending, limits, generated, strict=True):
            if not chunk.token_ids or len(chunk.token_ids) > limit:
                raise ValueError(
                    "Continuation made no progress or exceeded its allowance"
                )
            result[i] = Sample(
                result[i].text + chunk.text,
                result[i].token_ids + chunk.token_ids,
                chunk.finish_reason,
            )
            rounds[i] += 1
