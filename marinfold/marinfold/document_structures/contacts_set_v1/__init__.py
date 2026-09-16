# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-set-v1 structured residue-contact format."""

from .format import (
    COARSE_BINS_PER_SIGN,
    CONTACT_SLOTS,
    FINE_BINS,
    MAX_ABS_DELTA,
    ContactTarget,
    FixedResidueContacts,
    ResidueContactTarget,
    decode_contact,
    decode_delta,
    encode_delta,
    encode_residue_contacts,
    from_fixed_width,
    to_fixed_width,
)
from .loss import Assignment, target_anchored_slot_assignment

__all__ = [
    "COARSE_BINS_PER_SIGN",
    "CONTACT_SLOTS",
    "FINE_BINS",
    "MAX_ABS_DELTA",
    "Assignment",
    "ContactTarget",
    "FixedResidueContacts",
    "ResidueContactTarget",
    "decode_contact",
    "decode_delta",
    "encode_delta",
    "encode_residue_contacts",
    "from_fixed_width",
    "target_anchored_slot_assignment",
    "to_fixed_width",
]
