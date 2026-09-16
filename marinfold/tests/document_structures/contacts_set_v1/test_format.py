# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-set-v1 structured residue format."""

import pytest

from marinfold.document_structures.contacts_set_v1.format import (
    CONTACT_SLOTS,
    ContactTarget,
    FixedResidueContacts,
    decode_delta,
    encode_delta,
    encode_residue_contacts,
    from_fixed_width,
    to_fixed_width,
)


@pytest.mark.parametrize("delta", [-2048, -2047, -17, -16, -1, 1, 16, 17, 2047, 2048])
def test_delta_coarse_fine_roundtrip(delta: int):
    signed_coarse, fine = encode_delta(delta)
    assert decode_delta(signed_coarse, fine) == delta


@pytest.mark.parametrize("delta", [0, -2049, 2049])
def test_invalid_delta_rejected(delta: int):
    with pytest.raises(ValueError):
        encode_delta(delta)


@pytest.mark.parametrize("signed_coarse,fine", [(-1, 0), (256, 0), (0, -1), (0, 16)])
def test_invalid_coarse_fine_rejected(signed_coarse: int, fine: int):
    with pytest.raises(ValueError):
        decode_delta(signed_coarse, fine)


def test_residue_contacts_are_canonical_sorted_sets():
    target = encode_residue_contacts("ALA", [17, -31, 17], degrees=[0.2, 0.7, 0.9])

    assert target.aa == "ALA"
    assert [contact.delta for contact in target.contacts] == [-31, 17]
    # Exact duplicate deltas collapse and keep the strongest supplied degree.
    assert [contact.degree for contact in target.contacts] == [0.7, 0.9]


def test_multiple_contacts_in_same_coarse_bin_use_separate_records():
    target = encode_residue_contacts("GLY", [83, 90])

    assert [contact.delta for contact in target.contacts] == [83, 90]
    assert target.contacts[0].signed_coarse == target.contacts[1].signed_coarse
    assert target.contacts[0].fine != target.contacts[1].fine


def test_too_many_contacts_rejected():
    deltas = list(range(1, CONTACT_SLOTS + 2))

    with pytest.raises(ValueError, match="max supported"):
        encode_residue_contacts("SER", deltas)


def test_fixed_width_serialization_is_deterministic_and_padded():
    target = encode_residue_contacts("THR", [64, -9])
    fixed = to_fixed_width(target)

    assert fixed.aa == "THR"
    assert fixed.present[:2] == (True, True)
    assert fixed.present[2:] == (False,) * (CONTACT_SLOTS - 2)
    assert [decode_delta(c, f) for c, f in zip(fixed.signed_coarse[:2], fixed.fine[:2])] == [-9, 64]
    assert fixed.signed_coarse[2:] == (0,) * (CONTACT_SLOTS - 2)
    assert fixed.fine[2:] == (0,) * (CONTACT_SLOTS - 2)


def test_fixed_width_roundtrip_canonicalizes_slot_order():
    pos17 = ContactTarget(*encode_delta(17), degree=0.3)
    neg31 = ContactTarget(*encode_delta(-31), degree=0.8)
    fixed = FixedResidueContacts(
        aa="LYS",
        present=(True, True, *(False for _ in range(CONTACT_SLOTS - 2))),
        signed_coarse=(pos17.signed_coarse, neg31.signed_coarse, *(0 for _ in range(CONTACT_SLOTS - 2))),
        fine=(pos17.fine, neg31.fine, *(0 for _ in range(CONTACT_SLOTS - 2))),
        degree=(pos17.degree, neg31.degree, *(0.0 for _ in range(CONTACT_SLOTS - 2))),
    )

    target = from_fixed_width(fixed)

    assert [contact.delta for contact in target.contacts] == [-31, 17]
    assert [contact.degree for contact in target.contacts] == [0.8, 0.3]
