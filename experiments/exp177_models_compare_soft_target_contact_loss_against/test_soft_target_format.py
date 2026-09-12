from marinfold.document_structures.contacts_v1.generate import build_document
from marinfold.document_structures.contacts_v1.parse import AnalyzedStructure, RawContact, ResidueInfo, analyzed_to_row
from marinfold.document_structures.contacts_v1.vocab import BEGIN_STRUCTURE, CONTACT, END, POSITIONS, VOCABULARY

from premade_contacts_dataset import soft_target_contacts_v1_document_from_row
from preprocess_soft_targets import _row_from_document


def _toy_row():
    residues = tuple(
        ResidueInfo(seq_index=i, resname="ALA", resnum=i + 1, chain="A")
        for i in range(10)
    )
    contacts = (
        RawContact(seq_i=1, seq_j=7, degree=1.0),
        RawContact(seq_i=2, seq_j=8, degree=1.0),
    )
    analyzed = AnalyzedStructure(
        entry_id="toy",
        source_path="toy.cif",
        residues=residues,
        contacts=contacts,
        global_plddt=90.0,
    )
    return analyzed_to_row(analyzed), residues, contacts


def test_soft_target_uses_contacts_v1_position_tokens_not_sequence_indices():
    row, residues, contacts = _toy_row()
    generated = build_document("toy", residues, contacts, global_plddt=90.0)
    document = soft_target_contacts_v1_document_from_row(row)
    assert document is not None

    token_ids = list(document.token_ids)
    first_contact = token_ids.index(int(CONTACT))
    emitted = generated.contacts[0]
    expected_first, expected_second = (
        (emitted.pos_j, emitted.pos_i) if emitted.flipped else (emitted.pos_i, emitted.pos_j)
    )

    assert token_ids[first_contact + 1] == int(POSITIONS[expected_first])
    assert token_ids[first_contact + 2] == int(POSITIONS[expected_second])
    assert (token_ids[first_contact + 1] - int(POSITIONS[0]), token_ids[first_contact + 2] - int(POSITIONS[0])) != (
        emitted.seq_j if emitted.flipped else emitted.seq_i,
        emitted.seq_i if emitted.flipped else emitted.seq_j,
    )


def test_soft_target_prefix_matches_contacts_v1_generated_prefix():
    row, residues, contacts = _toy_row()
    generated = build_document("toy", residues, contacts, global_plddt=90.0)
    document = soft_target_contacts_v1_document_from_row(row)
    assert document is not None

    generated_prefix_text = generated.document.split()[: generated.document.split().index(BEGIN_STRUCTURE.text) + 1]
    expected_prefix_ids = [int(VOCABULARY.token(token)) for token in generated_prefix_text]
    token_ids = list(document.token_ids)

    assert token_ids[: len(expected_prefix_ids)] == expected_prefix_ids
    assert token_ids[len(expected_prefix_ids)] == int(CONTACT)


def test_precomputed_row_keeps_correct_contact_suffix_and_counts():
    row, residues, contacts = _toy_row()
    generated = build_document("toy", residues, contacts, global_plddt=90.0)
    document = soft_target_contacts_v1_document_from_row(row)
    assert document is not None

    out = _row_from_document(document, source_shard=0, slot_index=0, max_seq_len=128)
    first_contact = out["token_ids"].index(int(CONTACT))
    emitted = generated.contacts[0]
    expected_first, expected_second = (
        (emitted.pos_j, emitted.pos_i) if emitted.flipped else (emitted.pos_i, emitted.pos_j)
    )

    assert out["contact_count"] == len(generated.contacts)
    assert out["target_position_count"] == 3 * len(generated.contacts) + 1
    assert out["contact_first_ids"][0] == int(POSITIONS[expected_first])
    assert out["contact_second_ids"][0] == int(POSITIONS[expected_second])
    assert out["token_ids"][first_contact + 3 * len(generated.contacts)] == int(END)
