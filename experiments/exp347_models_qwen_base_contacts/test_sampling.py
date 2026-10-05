"""Continuation must retain each sampled prefix and terminate at the context limit."""

from sampling import Sample, finish_rollouts


def test_continuation_preserves_samples_and_uses_full_context() -> None:
    initial = [Sample("1 ", (1, 2), "length"), Sample("END", (9,), "stop")]
    calls = []

    def generate(
        prompts: list[list[int]], limits: list[int], seeds: list[int]
    ) -> list[Sample]:
        calls.append((prompts, limits, seeds))
        if len(calls) == 1:
            return [Sample("7\n", (3, 4), "length")]
        return [Sample("END", (5,), "stop")]

    result, rounds = finish_rollouts(initial, [[10], [11]], 2, 7, [4, 5], generate)
    assert calls[0][0] == [[10, 1, 2]]
    assert calls[1][0] == [[10, 1, 2, 3, 4]]
    assert calls[1][1] == [2]  # Remaining context, not an unbounded retry.
    assert calls[0][2] != calls[1][2]
    assert result == [Sample("1 7\nEND", (1, 2, 3, 4, 5), "stop"), initial[1]]
    assert rounds == [2, 0]
    assert initial[0].finish_reason == "length"


def test_exhausted_context_stays_invalid() -> None:
    def generate(
        prompts: list[list[int]], limits: list[int], seeds: list[int]
    ) -> list[Sample]:
        raise AssertionError("No room to generate")

    initial = [Sample("1 7", (1, 2), "length")]
    result, rounds = finish_rollouts(initial, [[10]], 2, 3, [4], generate)
    assert result == initial
    assert rounds == [0]
