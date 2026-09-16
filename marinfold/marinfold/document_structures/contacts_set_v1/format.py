# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure encoding helpers for contacts-set-v1.

contacts-set-v1 is a structured residue format: one record per residue, with
up to ``CONTACT_SLOTS`` unordered contacts represented as signed relative
sequence offsets.  The on-disk slot order is canonical only so the record has a
stable byte representation; model losses should treat the slots as a set.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MAX_ABS_DELTA = 2048
FINE_BINS = 16
COARSE_BINS_PER_SIGN = MAX_ABS_DELTA // FINE_BINS
CONTACT_SLOTS = 16


@dataclass(frozen=True, order=True)
class ContactTarget:
    """One contact offset in hierarchical coarse/fine form.

    ``signed_coarse`` is in ``[0, 255]`` for the default constants.  Negative
    offsets use bins ``0..127``; positive offsets use bins ``128..255``.
    ``fine`` is the offset within that coarse bin.
    """

    signed_coarse: int
    fine: int
    degree: float = 1.0

    @property
    def delta(self) -> int:
        """Signed sequence offset represented by this target."""
        return decode_delta(self.signed_coarse, self.fine)


@dataclass(frozen=True)
class ResidueContactTarget:
    """Logical contacts-set-v1 target for one residue."""

    aa: str
    contacts: tuple[ContactTarget, ...]


@dataclass(frozen=True)
class FixedResidueContacts:
    """Fixed-width serialization arrays for one residue.

    Padding rows have ``present=False``.  Their coarse/fine values are ignored.
    """

    aa: str
    present: tuple[bool, ...]
    signed_coarse: tuple[int, ...]
    fine: tuple[int, ...]
    degree: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.present), len(self.signed_coarse), len(self.fine), len(self.degree)}
        if lengths != {CONTACT_SLOTS}:
            raise ValueError(f"fixed residue arrays must all have length {CONTACT_SLOTS}, got {sorted(lengths)}")


def encode_delta(delta: int) -> tuple[int, int]:
    """Encode a nonzero signed relative offset as ``(signed_coarse, fine)``."""
    if delta == 0:
        raise ValueError("delta=0 is a self-contact and cannot be encoded")
    abs_delta = abs(delta)
    if abs_delta > MAX_ABS_DELTA:
        raise ValueError(f"abs(delta) must be <= {MAX_ABS_DELTA}, got {delta}")

    abs0 = abs_delta - 1
    coarse = abs0 // FINE_BINS
    fine = abs0 % FINE_BINS
    signed_coarse = coarse + (COARSE_BINS_PER_SIGN if delta > 0 else 0)
    return signed_coarse, fine


def decode_delta(signed_coarse: int, fine: int) -> int:
    """Decode ``(signed_coarse, fine)`` back to a signed relative offset."""
    if not 0 <= signed_coarse < 2 * COARSE_BINS_PER_SIGN:
        raise ValueError(f"signed_coarse out of range: {signed_coarse}")
    if not 0 <= fine < FINE_BINS:
        raise ValueError(f"fine out of range: {fine}")

    is_positive = signed_coarse >= COARSE_BINS_PER_SIGN
    coarse = signed_coarse % COARSE_BINS_PER_SIGN
    abs_delta = coarse * FINE_BINS + fine + 1
    return abs_delta if is_positive else -abs_delta


def decode_contact(contact: ContactTarget) -> int:
    """Return the signed offset represented by ``contact``."""
    return decode_delta(contact.signed_coarse, contact.fine)


def encode_residue_contacts(
    aa: str,
    deltas: Iterable[int],
    *,
    degrees: Sequence[float] | None = None,
    max_contacts: int = CONTACT_SLOTS,
) -> ResidueContactTarget:
    """Build a canonical logical residue target from signed contact offsets.

    Exact duplicate deltas are collapsed, preserving the largest supplied degree
    for that offset.  The returned contacts are sorted by signed delta solely for
    deterministic serialization; callers should treat them as an unordered set.
    """
    deltas_tuple = tuple(int(delta) for delta in deltas)
    if degrees is not None and len(degrees) != len(deltas_tuple):
        raise ValueError("degrees must have the same length as deltas")

    degree_by_delta: dict[int, float] = {}
    for idx, delta in enumerate(deltas_tuple):
        # Validate even if an earlier duplicate already set this delta.
        encode_delta(delta)
        degree = float(degrees[idx]) if degrees is not None else 1.0
        degree_by_delta[delta] = max(degree_by_delta.get(delta, float("-inf")), degree)

    if len(degree_by_delta) > max_contacts:
        raise ValueError(f"residue has {len(degree_by_delta)} contacts; max supported is {max_contacts}")

    contacts = []
    for delta in sorted(degree_by_delta):
        signed_coarse, fine = encode_delta(delta)
        contacts.append(ContactTarget(signed_coarse=signed_coarse, fine=fine, degree=degree_by_delta[delta]))
    return ResidueContactTarget(aa=aa, contacts=tuple(contacts))


def to_fixed_width(target: ResidueContactTarget) -> FixedResidueContacts:
    """Convert a logical residue target to fixed-width padded arrays."""
    if len(target.contacts) > CONTACT_SLOTS:
        raise ValueError(f"residue has {len(target.contacts)} contacts; max supported is {CONTACT_SLOTS}")

    present = [False] * CONTACT_SLOTS
    signed_coarse = [0] * CONTACT_SLOTS
    fine = [0] * CONTACT_SLOTS
    degree = [0.0] * CONTACT_SLOTS
    for idx, contact in enumerate(target.contacts):
        present[idx] = True
        signed_coarse[idx] = contact.signed_coarse
        fine[idx] = contact.fine
        degree[idx] = contact.degree
    return FixedResidueContacts(
        aa=target.aa,
        present=tuple(present),
        signed_coarse=tuple(signed_coarse),
        fine=tuple(fine),
        degree=tuple(degree),
    )


def from_fixed_width(fixed: FixedResidueContacts) -> ResidueContactTarget:
    """Convert fixed-width padded arrays back to a canonical logical target."""
    contacts = [
        ContactTarget(signed_coarse=coarse, fine=fine, degree=degree)
        for is_present, coarse, fine, degree in zip(fixed.present, fixed.signed_coarse, fixed.fine, fixed.degree)
        if is_present
    ]
    by_delta = {contact.delta: contact.degree for contact in contacts}
    return encode_residue_contacts(fixed.aa, by_delta.keys(), degrees=tuple(by_delta.values()))
