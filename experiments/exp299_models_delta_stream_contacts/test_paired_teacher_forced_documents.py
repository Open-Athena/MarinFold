from build_paired_teacher_forced_documents import (
    contacts_v1_document,
    delta_v2_document,
)
from compute_contacts_delta_stream_documents import (
    CONTACTS_BEGIN_TOKEN_ID,
    DOC_END_TOKEN_ID,
)


def test_paired_documents_preserve_sequence_and_contacts() -> None:
    sequence = "ACDEFGHIK"
    contacts = [[0, 8, 0.5], [1, 4, 0.9], [1, 7, 0.0005]]

    contacts_v1 = contacts_v1_document("test", "protein", sequence, contacts)
    delta_v2 = delta_v2_document(sequence, contacts)

    assert contacts_v1[0] == 2
    assert contacts_v1[-1] == 1
    assert contacts_v1.count(5) == 1
    assert delta_v2[-1] == DOC_END_TOKEN_ID
    assert delta_v2.index(CONTACTS_BEGIN_TOKEN_ID) == len(sequence) + 1
    # The one eligible undirected pair is emitted in both residue segments.
    assert len(delta_v2) == 1 + len(sequence) + 1 + len(sequence) + 2 + 1


def test_unknown_amino_acid_uses_reserved_delta_token() -> None:
    token_ids = delta_v2_document("AX", [])
    assert token_ids[1:3] == [1, 24]
