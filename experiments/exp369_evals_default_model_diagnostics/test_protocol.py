"""Scientific invariants: prompts cannot score themselves or corrupt coordinates."""

import numpy as np

from analyze_cases import exclusive_preference
from protocol import contact_order, coverage_milestones, remaining_metrics
from spiking_worker import make_prompt, parsed_contacts


def test_prompt_copying_cannot_increase_remaining_precision() -> None:
    truth = [[0, 7], [1, 8]]
    supplied = [[0, 7]]
    scores = np.zeros((10, 10), dtype=np.uint16)
    scores[0, 7] = 100
    scores[0, 6] = 20
    scores[1, 8] = 10
    result = remaining_metrics(scores, truth, list(range(10)), supplied)
    assert result["n_remaining_true"] == 1
    assert result["r_precision"] == 0
    scores[1, 8] = 21
    assert remaining_metrics(scores, truth, list(range(10)), supplied)["r_precision"] == 1


def test_unresolved_and_range_contacts_are_removed_from_r() -> None:
    scores = np.zeros((30, 30))
    scores[0, 29] = 100
    scores[0, 25] = 1
    truth = [[0, 8], [0, 25], [0, 29]]
    result = remaining_metrics(scores, truth, list(range(29)), [], min_separation=24)
    assert result["n_remaining_true"] == 1
    assert result["r_precision"] == 1
    assert remaining_metrics(scores, truth, list(range(29)), [[0, 25]], 24)["r_precision"] is None


def test_sse_coverage_precedes_coil_only_contacts() -> None:
    elements = [-1] * 30
    elements[0] = elements[1] = 0
    elements[10] = elements[11] = 1
    pairs = [[0, 20], [1, 21], [10, 22], [11, 23], [24, 29]]
    for seed in range(20):
        order = contact_order(pairs, elements, "sse_balanced", str(seed))
        assert len(order) == len(pairs)
        assert {tuple(pair) for pair in order} == {tuple(pair) for pair in pairs}
        assert {elements[pair[0]] for pair in order[:2]} == {0, 1}
        assert order[-1] == [24, 29]


def test_coverage_milestones_count_both_endpoints_and_saturate() -> None:
    elements = [-1] * 30
    elements[0], elements[10], elements[20] = 0, 1, 2
    order = [[0, 10], [0, 20], [10, 20], [24, 29]]
    milestones = coverage_milestones(order, elements)
    assert milestones == [{"axis": "sse_round", "dose": 1, "k": 2},
                           {"axis": "sse_round", "dose": 2, "k": 3},
                           {"axis": "sse_round", "dose": 4, "k": 3}]


def test_prompt_realizations_preserve_supplied_sequence_pairs() -> None:
    target = {"stem": "mapping_test", "L": 90, "sequence": "ACDEFGHIKLMNPQRSTVWY" * 4 + "ACDEFGHIKL"}
    pairs = [[0, 89], [1, 75], [3, 35]]
    prefixes = set()
    for rollout in range(20):
        prompt, inverse = make_prompt(target, pairs, 0, rollout)
        restored, diagnostics = parsed_contacts(prompt, inverse)
        assert restored == pairs
        assert not any(diagnostics.values())
        prefixes.add(prompt.split("<begin_statements>")[0])
    assert len(prefixes) == 20


def test_retracted_contact_does_not_vote() -> None:
    text = "<begin_statements> <contact> <p1999> <p8> <retract> <p8> <p1999> <contact> <p0> <p9> <end>"
    pairs, diagnostics = parsed_contacts(text, {1999: 0, 8: 9, 0: 1, 9: 10})
    assert pairs == [[1, 10]]
    assert not any(diagnostics.values())


def test_nonexistent_residue_is_counted_without_discarding_valid_pairs() -> None:
    text = "<contact> <p431> <p468> <contact> <p451> <p469> <end>"
    inverse = {position: position - 431 for position in range(431, 469)}
    pairs, diagnostics = parsed_contacts(text, inverse)
    assert pairs == [[0, 37]]
    assert diagnostics == {"malformed": 0, "out_of_range": 1, "too_close": 0}


def test_exclusive_preference_is_size_balanced_and_tie_aware() -> None:
    votes = np.zeros((12, 12))
    votes[0, 8] = 10
    votes[0, 9] = 20
    votes[1, 10] = 20
    assert exclusive_preference(votes, [[0, 8], [0, 9]], [[1, 10]]) == .75
    assert exclusive_preference(votes, [[1, 10]], [[0, 8], [0, 9]]) == .25
