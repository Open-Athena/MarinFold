"""Check information preservation across randomized cyclic indexing."""

import pytest
from common import convert_document, split_key

TEXT = (
    "<contacts-v1> <begin_sequence> <p0> <ASP> <p1998> <ALA> "
    "<c-term> <p4> <p2> <GLY> <p1999> <CYS> <p4> <LEU> "
    "<n-term> <p1998> <p1> <GLU> <p3> <HIS> "
    "<begin_statements> <contact> <p4> <p1998> <end>"
)


def test_wraparound_and_order_are_preserved() -> None:
    doc = convert_document(TEXT)
    assert doc.sequence == "ACDEGHL"
    assert doc.contacts == ((7, 1),)
    assert doc.prompted_completion == "7 1\nEND\n"
    assert doc.raw_prefix + doc.raw_completion == TEXT


def test_source_cluster_is_not_split_between_train_and_validation() -> None:
    assert split_key("AAAA", "cluster-1") == split_key("CCCC", "cluster-1")
    assert split_key("AAAA", "cluster-1") != split_key("AAAA", "cluster-2")


@pytest.mark.parametrize(
    "text",
    [
        TEXT.replace("<p0> <ASP>", "<p1998> <ASP>"),
        TEXT.replace("<c-term> <p4>", "<c-term> <p3>"),
        TEXT.replace("<contact> <p4>", "<contact> <p3>"),
        TEXT.replace("<end>", "<contact> <p1998> <p4> <end>"),
        TEXT.replace("<contact>", "<retract>"),
    ],
)
def test_invalid_inputs_fail_loudly(text: str) -> None:
    with pytest.raises((ValueError, KeyError)):
        convert_document(text)
