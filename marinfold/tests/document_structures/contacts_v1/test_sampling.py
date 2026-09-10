# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise EOS gating against a scripted sampler that prefers early EOS."""

import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import sample_contacts
from marinfold.document_structures.contacts_v1.vocab import all_domain_tokens


class ScriptedBackend:
    def __init__(self, prompts: list[str], streams: list[str], *, early_end: bool = True):
        self.tokenizer = build_tokenizer(all_domain_tokens())
        self.prompts = [self.encode(prompt) for prompt in prompts]
        self.streams = [self.encode(stream) for stream in streams]
        self.early_end = early_end
        self.calls: list[dict] = []

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def sample_completions(self, prompts: list[list[int]], **kwargs) -> list[list[int]]:
        self.calls.append(kwargs)
        minimum = kwargs.get("min_new_tokens", 0)
        maximum = kwargs["max_new_tokens"]
        outputs = []
        for prompt in prompts:
            row = next(i for i, base in enumerate(self.prompts) if prompt[:len(base)] == base)
            offset = len(prompt) - len(self.prompts[row])
            if not minimum and self.early_end:
                outputs.append([])
                continue
            output = self.streams[row][offset:offset + maximum]
            stop_id = kwargs["stop_token_id"]
            if stop_id in output:
                output = output[:output.index(stop_id)]
            outputs.append(output)
        return outputs


def test_thirty_new_contacts_excludes_ten_prompt_contacts() -> None:
    contact = "<contact> <p0> <p10> "
    backend = ScriptedBackend(["<begin_statements> " + contact * 10], [contact * 40])
    before = [list(prompt) for prompt in backend.prompts]
    result = sample_contacts(
        backend, backend.prompts, max_new_tokens=150, min_new_contacts=30,
    )
    assert result == [backend.encode(contact * 30)]
    assert backend.prompts == before
    assert [call["min_new_tokens"] for call in backend.calls] == [90, 0]


@pytest.mark.parametrize("minimum", [None, 0])
def test_default_allows_immediate_end_in_one_unchanged_call(minimum: int | None) -> None:
    backend = ScriptedBackend(["<begin_statements>"], ["<contact> <p0> <p10>"])
    assert sample_contacts(
        backend, backend.prompts, max_new_tokens=20, min_new_contacts=minimum, seed=42,
    ) == [[]]
    assert len(backend.calls) == 1
    assert "min_new_tokens" not in backend.calls[0]
    assert backend.calls[0]["seed"] == 42


def test_think_malformed_and_partial_contacts_have_independent_row_budgets() -> None:
    streams = [
        "<think> <contact> <p0> <p10> <contact> <p1> <p11>",
        "<contact> <think> <p0> <retract> <p0> <p10> "
        "<contact> <p2> <p12> <contact> <p3> <p13>",
    ]
    backend = ScriptedBackend(["<p0> <begin_statements>", "<p1> <begin_statements>"], streams)
    result = sample_contacts(backend, backend.prompts, max_new_tokens=30, min_new_contacts=2)
    assert result == [backend.encode(stream) for stream in streams]


def test_minimum_does_not_force_stopping_at_budget() -> None:
    stream = "<contact> <p0> <p10> <contact> <p1> <p11> <end>"
    backend = ScriptedBackend(["<begin_statements>"], [stream], early_end=False)
    result = sample_contacts(backend, backend.prompts, max_new_tokens=20, min_new_contacts=1)
    assert result == [backend.encode(stream)[:-1]]


@pytest.mark.parametrize("minimum", [-1, 1.5])
def test_invalid_minimum_rejected_before_sampling(minimum: int) -> None:
    backend = ScriptedBackend(["<begin_statements>"], [""])
    with pytest.raises(ValueError, match="non-negative integer"):
        sample_contacts(backend, backend.prompts, max_new_tokens=20, min_new_contacts=minimum)
    assert not backend.calls


def test_impossible_token_budget_rejected_before_sampling() -> None:
    backend = ScriptedBackend(["<begin_statements>"], [""])
    with pytest.raises(ValueError, match="at least 90"):
        sample_contacts(backend, backend.prompts, max_new_tokens=89, min_new_contacts=30)
    assert not backend.calls


def test_token_cap_reports_unmet_minimum() -> None:
    backend = ScriptedBackend(["<begin_statements>"], ["<think> " * 12])
    with pytest.raises(RuntimeError, match="0/2 new contacts"):
        sample_contacts(backend, backend.prompts, max_new_tokens=12, min_new_contacts=2)


def test_minimum_can_exactly_fill_token_cap() -> None:
    stream = "<contact> <p0> <p10>"
    backend = ScriptedBackend(["<begin_statements>"], [stream])
    assert sample_contacts(
        backend, backend.prompts, max_new_tokens=3, min_new_contacts=1,
    ) == [backend.encode(stream)]


def test_backend_early_stop_is_an_error() -> None:
    backend = ScriptedBackend(["<begin_statements>"], ["<end>"])
    with pytest.raises(RuntimeError, match="despite EOS suppression"):
        sample_contacts(backend, backend.prompts, max_new_tokens=10, min_new_contacts=1)
