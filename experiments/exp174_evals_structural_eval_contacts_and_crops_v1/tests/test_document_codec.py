# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The decode layer: digit arithmetic, the crop state machine, the estimator.

The load-bearing test is :func:`test_round_trip_recovers_ground_truth`, which
runs the format's **own generator** on a synthetic structure and checks the
decoder gets the coordinates back. The generator knows nothing about the
decoder and places the structure in a random rotated + translated frame, so a
sign error, an off-by-one in the digit split, or a mis-counted crop header
cannot cancel out.
"""

import math
import random

import numpy as np
import pytest

from document_codec import (
    CoordinateEstimate,
    Observation,
    box_center,
    box_from_header,
    crop_header,
    pass1_budget,
    parse_observations,
    position_from_crop,
    render_crop,
    sequence_prefix,
    start_index,
    synthesize_pass1,
    xyz_digits,
)
from marinfold.document_structures.contacts_and_crops_v1 import build_document
from marinfold.document_structures.contacts_v1.parse import residues_from_sequence

_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"


def _synthetic_atoms(sequence: str, seed: int = 0):
    """A plausible little chain: CA/N/C/O on a slowly turning helix-ish path."""
    rng = random.Random(seed)
    atoms = {}
    for k in range(len(sequence)):
        angle = 0.35 * k
        base = np.array(
            [12.0 * math.cos(angle), 12.0 * math.sin(angle), 1.5 * k], dtype=float
        )
        entries = []
        for name, offset in (
            ("N", (-1.2, 0.0, 0.0)),
            ("CA", (0.0, 0.0, 0.0)),
            ("C", (1.2, 0.3, 0.0)),
            ("O", (1.6, 1.4, 0.0)),
        ):
            jitter = np.array([rng.gauss(0, 0.05) for _ in range(3)])
            x, y, z = base + np.array(offset) + jitter
            entries.append((name, float(x), float(y), float(z)))
        atoms[k] = tuple(entries)
    return atoms


def test_digit_arithmetic_matches_the_spec_worked_example():
    # SPEC → "Worked digit example": header <xyz-200> <xyz-070> names the cell
    # x∈[200,210) y∈[70,80) z∈[0,10); body <xyz-526> <xyz-631> is (205.6, 72.3, 6.1).
    cell = box_from_header("<xyz-200>", "<xyz-070>")
    assert cell == (20, 7, 0)
    position = position_from_crop(cell, "<xyz-526>", "<xyz-631>")
    assert np.allclose(position, [205.6, 72.3, 6.1])


def test_xyz_digits_unpacks_the_triple():
    assert xyz_digits("<xyz-000>") == (0, 0, 0)
    assert xyz_digits("<xyz-526>") == (5, 2, 6)
    assert xyz_digits("<xyz-999>") == (9, 9, 9)
    assert xyz_digits("<CA>") is None


def test_box_center_is_the_midpoint_of_the_cell():
    # Cell c holds v with round(v*10) in [100c, 100c+99] -> v in [10c-0.05, 10c+9.95].
    assert np.allclose(box_center((0, 0, 0)), [4.95, 4.95, 4.95])
    assert np.allclose(box_center((20, 7, 0)), [204.95, 74.95, 4.95])


def test_crop_header_round_trips_through_the_decoder():
    for cell in [(0, 0, 0), (20, 7, 0), (99, 99, 99), (3, 41, 59)]:
        tokens = crop_header(cell)
        assert tokens[0] == "<crop>"
        assert box_from_header(tokens[1], tokens[2]) == cell


def test_parse_counts_visit_index_per_box():
    # Two crops on the same box, then one on another: visit indices 0, 1, 0.
    start = 0
    tokens = []
    for cell in [(1, 1, 1), (1, 1, 1), (2, 2, 2)]:
        tokens += crop_header(cell) + ["<p0>", "<CA>", "<xyz-000>", "<xyz-000>"]
    observations = list(parse_observations(tokens, start=start, length=4))
    assert [o.visit_index for o in observations] == [0, 1, 0]
    # σ = 1/(i+1)^2, so the variance falls steeply with the visit index.
    assert observations[0].variance > observations[1].variance


def test_estimator_is_precision_weighted():
    estimate = CoordinateEstimate()
    # A coarse observation at the origin and a sharp one at (10, 0, 0): the
    # posterior must sit essentially on the sharp one.
    estimate.add(Observation(0, "CA", np.zeros(3), variance=100.0, source="pass1", visit_index=-1))
    estimate.add(
        Observation(0, "CA", np.array([10.0, 0, 0]), variance=0.01, source="crop", visit_index=3)
    )
    assert estimate.position((0, "CA"))[0] == pytest.approx(10.0, abs=1e-2)
    assert estimate.sigma((0, "CA")) < 0.11
    assert estimate.refined_keys() == {(0, "CA")}


def test_round_trip_recovers_ground_truth():
    """Generate a real document from known coordinates and decode it back.

    The end-to-end check on the whole decode layer. Tolerance is set by the
    format, not by the decoder: Pass-1 mentions localize an atom to a 10 Å box,
    and Pass-2 crops start at σ=1 Å and sharpen, so a *refined* atom should come
    back within about an ångström while a box-only atom is worth ±5 Å.
    """
    residues = residues_from_sequence(_SEQUENCE)
    atoms = _synthetic_atoms(_SEQUENCE)
    result = build_document("round-trip-test", residues, [], atoms)
    assert result is not None

    estimate = CoordinateEstimate()
    estimate.add_all(
        parse_observations(
            result.document.split(),
            start=result.start_index,
            length=len(_SEQUENCE),
        )
    )
    assert len(estimate) > 0

    # The document places the structure in a random rotated + translated frame,
    # so compare after a Kabsch fit — exactly what the scorer does.
    from biotite.structure import AtomArray, rmsd, superimpose

    keys = sorted(estimate.refined_keys())
    assert len(keys) >= 10, "expected the fine reserve to refine some atoms"

    truth = np.stack([
        np.array([c for n, *c in atoms[k] if n == name][0]) for k, name in keys
    ])
    decoded = np.stack([estimate.position(key) for key in keys])

    reference = AtomArray(len(keys))
    reference.coord = truth.astype(np.float32)
    subject = AtomArray(len(keys))
    subject.coord = decoded.astype(np.float32)
    fitted, _ = superimpose(reference, subject)
    assert float(rmsd(reference, fitted)) < 1.5


def test_synthesize_pass1_fills_the_cap_and_decodes_back():
    """Plan F's global feedback channel must produce a legal Pass-1 section.

    The model never saw a *short* Pass-1 section — the format always fills the
    cap — so a synthesized one that stopped early would be off-distribution.
    """
    estimate = CoordinateEstimate()
    for k in range(20):
        for name in ("N", "CA", "C", "O"):
            estimate.add(
                Observation(
                    k, name, np.array([100.0 + k, 200.0, 300.0]),
                    variance=0.01, source="crop", visit_index=3,
                )
            )
    cap, _ = pass1_budget(20)
    tokens = synthesize_pass1(
        estimate, start=7, cap_tokens=cap, rng=random.Random(0), noise_sigma=0.0
    )
    assert cap - len(tokens) < 4, "Pass 1 must fill its cap"
    assert len(tokens) % 4 == 0

    decoded = list(parse_observations(tokens, start=7, length=20))
    assert len(decoded) == len(tokens) // 4
    assert all(o.source == "pass1" for o in decoded)
    # Noise-free synthesis: every atom must land in its own 10 Å box.
    for observation in decoded:
        truth = estimate.position(observation.key)
        assert np.all(np.abs(observation.position - truth) <= 5.0)


def test_render_crop_round_trips_at_tenths_resolution():
    estimate = CoordinateEstimate()
    positions = {}
    for k in range(5):
        position = np.array([123.4 + k * 0.3, 45.6, 78.9])
        positions[(k, "CA")] = position
        estimate.add(
            Observation(k, "CA", position, variance=0.01, source="crop", visit_index=3)
        )
    cell = (12, 4, 7)
    tokens = render_crop(cell, sorted(positions), estimate, start=0)
    decoded = {
        o.key: o.position
        for o in parse_observations(tokens, start=0, length=5)
        if o.source == "crop"
    }
    assert set(decoded) == set(positions)
    for key, position in positions.items():
        assert np.allclose(decoded[key], np.round(position * 10) / 10, atol=1e-9)


def test_sequence_prefix_is_the_documents_own_sequence_section():
    prefix = sequence_prefix("prefix-test", _SEQUENCE)
    assert prefix is not None
    assert prefix[0] == "<contacts-and-crops-v1>"
    assert prefix[1] == "<begin_sequence>"
    assert prefix[-1] == "<begin_statements>"
    # 2 framing + 2 per residue + 4 terminus tokens + <begin_statements>.
    assert len(prefix) == 2 + 2 * len(_SEQUENCE) + 4 + 1

    # And it agrees with what a full document for the same id emits, because the
    # start index and the shuffle are RNG draws 1 and 2 — before the frame draws
    # a coordinate-free call skips.
    residues = residues_from_sequence(_SEQUENCE)
    full = build_document("prefix-test", residues, [], _synthetic_atoms(_SEQUENCE))
    tokens = full.document.split()
    assert tokens[: len(prefix)] == prefix
    assert start_index("prefix-test", _SEQUENCE) == full.start_index
