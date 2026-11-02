# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from common import parse_rollout
from conditioning_worker import prompt
from prepare_conditioning import contexts_for_target, range_bin


def protocol_record() -> dict:
    pairs = [(i, j, 1.0) for i in range(5, 20) for j in range(i + 6, 60)]
    return {
        "stem": "protocol-test",
        "L": 60,
        "contacts": pairs[:80],
        "resolved": list(range(60)),
    }


def test_predicted_context_selection_does_not_use_truth_or_resolved_mask():
    first = protocol_record()
    alternatives = [(i, j, 1.0) for i in range(15, 25) for j in range(i + 6, 60)]
    second = dict(first, contacts=alternatives[:80], resolved=list(range(1, 60)))
    votes = np.zeros((60, 60), dtype=np.int16)
    votes[0, 59] = votes[59, 0] = 100
    contexts_a = contexts_for_target(first, votes, 0)
    contexts_b = contexts_for_target(second, votes, 0)
    assert contexts_a["pred_large"] == contexts_b["pred_large"]
    assert contexts_a["pred_small"] == contexts_b["pred_small"]
    # This top prediction deliberately includes a residue omitted from the
    # second record's resolved mask and is false under both reference maps.
    assert contexts_b["pred_large"][0] == (0, 59)
    assert len(contexts_b["pred_large"]) == 60 // 3


def test_true_false_controls_are_nested_distinct_and_separation_matched():
    record = protocol_record()
    votes = np.zeros((60, 60), dtype=np.int16)
    contexts = contexts_for_target(record, votes, 0)
    truth = {(i, j) for i, j, _ in record["contacts"]}
    for kind in ("true", "false", "pred"):
        assert contexts[f"{kind}_small"] == contexts[f"{kind}_large"][:10]
        assert len(set(contexts[f"{kind}_large"])) == 20
    assert set(contexts["true_large"]) <= truth
    assert not (set(contexts["false_large"]) & truth)
    assert [range_bin(j - i) for i, j in contexts["true_large"]] == [
        range_bin(j - i) for i, j in contexts["false_large"]
    ]
    other = contexts_for_target(record, votes, 1)
    assert other["true_large"] != contexts["true_large"]
    assert other["pred_large"] == contexts["pred_large"]


def test_fixed_context_survives_100_position_remappings_and_order_shuffles():
    given = [[0, 6], [5, 15], [15, 39]]
    original = [pair.copy() for pair in given]
    serialized = set()
    for rollout in range(100):
        positions = [(1990 + rollout * 37 + residue) % 2000 for residue in range(40)]
        text = prompt(
            "<begin_statements>", positions, given, f"rep0:pred_large:{rollout}"
        )
        recovered = parse_rollout(text, {p: i for i, p in enumerate(positions)})
        assert set(recovered) == set(map(tuple, given))
        serialized.add(tuple(recovered))
    assert len(serialized) > 1
    assert given == original


def test_no_contact_prompt_is_exactly_the_shared_realization_prefix():
    prefix = "<begin_sequence> A C <end_sequence> <begin_statements>"
    assert prompt(prefix, [1999, 0], [], "iid") == prefix
    assert prompt(prefix, [1999, 0], [], "iid_repeat") == prefix
