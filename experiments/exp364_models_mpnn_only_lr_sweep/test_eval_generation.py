"""Check prefix preservation, sample identity and context bounds on continuation."""

from dataclasses import dataclass, replace

import pytest

from experiments.exp364_models_mpnn_only_lr_sweep.eval_generation import (
    Completion,
    Rollout,
    extend_capped_rollouts,
)


@dataclass
class Parameters:
    seed: int
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1

    def clone(self) -> "Parameters":
        return replace(self)


class Tokenizer:
    def decode(self, tokens: tuple[int, ...], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return " ".join(str(token) for token in tokens)


class Engine:
    def __init__(
        self, *, change_prefix: bool = False, finish_reason: str = "stop"
    ) -> None:
        self.calls = []
        self.change_prefix = change_prefix
        self.finish_reason = finish_reason

    def generate(
        self, prompts: list[dict], parameters: list[Parameters], *, use_tqdm: bool
    ) -> list[Rollout]:
        self.calls.append((prompts, parameters))
        return [
            Rollout(
                tuple(prompt["prompt_token_ids"])
                + ((999,) if self.change_prefix else ()),
                (Completion((11, 12), "11 12", self.finish_reason),),
            )
            for prompt in prompts
        ]


def test_only_capped_sample_continues_and_preserves_its_entire_prefix() -> None:
    stopped = Rollout((1, 2), (Completion((7, 8), "7 8", "stop"),))
    capped = Rollout((3, 4), (Completion((9, 10), "9 10", "length"),))
    original = [Parameters(20, 2), Parameters(21, 2)]
    engine = Engine()
    outputs, records = extend_capped_rollouts(
        engine, [stopped, capped], original, Tokenizer(), max_context=10
    )
    assert outputs[0] == stopped
    assert outputs[1].outputs[0] == Completion((9, 10, 11, 12), "9 10 11 12", "stop")
    assert capped.outputs[0].token_ids == (9, 10)
    prompts, parameters = engine.calls[0]
    assert prompts == [{"prompt_token_ids": [3, 4, 9, 10]}]
    assert parameters[0].max_tokens == 6
    assert parameters[0].seed != original[1].seed
    assert (parameters[0].temperature, parameters[0].top_p, parameters[0].top_k) == (
        1.0,
        0.95,
        -1,
    )
    assert original[1] == Parameters(21, 2)
    assert records[0]["sample_index"] == 1
    assert records[0]["total_tokens"] == 4
    assert records[0]["added_tokens"] == 2
    repeated = Engine()
    _, repeated_records = extend_capped_rollouts(
        repeated, [stopped, capped], original, Tokenizer(), max_context=10
    )
    assert repeated_records == records


@pytest.mark.parametrize("finish_reason,context", [("stop", 8), ("length", 4)])
def test_completed_or_context_exhausted_samples_are_never_redrawn(
    finish_reason: str, context: int
) -> None:
    sample = Rollout((1, 2), (Completion((3, 4), "3 4", finish_reason),))
    engine = Engine()
    outputs, records = extend_capped_rollouts(
        engine, [sample], [Parameters(0, 2)], Tokenizer(), max_context=context
    )
    assert not engine.calls
    assert outputs == [sample]
    if finish_reason == "length":
        assert records[0]["finish_reason"] == "length"
        assert records[0]["additional_budget"] == 0


def test_engine_cannot_replace_the_conditioning_prefix() -> None:
    sample = Rollout((1,), (Completion((3, 4), "3 4", "length"),))
    with pytest.raises(ValueError, match="preserved sample prefix"):
        extend_capped_rollouts(
            Engine(change_prefix=True), [sample], [Parameters(0, 2)], Tokenizer()
        )


def test_continuation_that_hits_context_limit_remains_incomplete() -> None:
    sample = Rollout((1, 2), (Completion((3, 4), "3 4", "length"),))
    outputs, records = extend_capped_rollouts(
        Engine(finish_reason="length"),
        [sample],
        [Parameters(0, 2)],
        Tokenizer(),
        max_context=6,
    )
    assert outputs[0].outputs[0].finish_reason == "length"
    assert records[0]["total_tokens"] == 4
    assert records[0]["finish_reason"] == "length"


def test_engine_aborts_are_not_treated_as_token_caps() -> None:
    sample = Rollout((1,), (Completion((3,), "3", "abort"),))
    with pytest.raises(ValueError, match="Unexpected finish reason"):
        extend_capped_rollouts(Engine(), [sample], [Parameters(0, 2)], Tokenizer())
