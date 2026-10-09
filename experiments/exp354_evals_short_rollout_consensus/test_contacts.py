from contacts import limits, sequence_pairs, snapshots


def test_cutoff_preserves_state_before_later_retract() -> None:
    tokens = "<contact> <p1998> <p8> <retract> <p8> <p1998> <contact> <p1999> <p9> <end>".split()
    result, count = snapshots(tokens, [1, 2, 5])
    assert count == 2
    assert result[1] == {"pairs": frozenset({(8, 1998)}), "tokens": 3, "emitted": 1}
    assert result[2]["pairs"] == frozenset({(9, 1999)})
    assert result[5]["tokens"] == 10
    assert sequence_pairs(result[1]["pairs"], 1995, 20) == {(3, 13)}


def test_duplicate_and_invalid_contacts_spend_budget_but_not_extra_votes() -> None:
    tokens = "<contact> <p0> <p10> <contact> <p10> <p0> <contact> <p0> <p100>".split()
    result, count = snapshots(tokens, [2, 3])
    assert count == 3
    assert sequence_pairs(result[3]["pairs"], 0, 20) == {(0, 10)}
    assert result[2]["tokens"] == 6


def test_partial_malformed_and_eos_do_not_fabricate_contacts() -> None:
    tokens = "<think> <contact> <p0> <think> <contact> <p0> <p8> <end> <contact> <p1> <p9>".split()
    result, count = snapshots(tokens, [2])
    assert count == 1
    assert result[2]["tokens"] == 8
    assert result[2]["pairs"] == frozenset({(0, 8)})
    assert snapshots(["<contact>", "<p0>"], [1])[1] == 0
    assert limits(47) == {"5": 5, "10": 10, "20": 20, "L5": 9, "L2": 23}
