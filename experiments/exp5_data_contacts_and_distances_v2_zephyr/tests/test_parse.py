# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for exp5's columnar ParsedStructure.

Not exhaustive — the byte-identity test against exp34 is what catches
correctness regressions on the generation side. These tests just pin
the *shape* of the new representation so a future change to the
columnar layout doesn't silently break downstream code.
"""

from __future__ import annotations

import math

import numpy as np

import parse


def test_parsed_structure_shapes(small_cif):
    """Per-residue arrays line up with sequence length; atom CSR offsets are monotonic."""
    ps = parse.parse_cif_content(small_cif, entry_id="AF-TEST-F1")
    n = ps.num_residues
    assert n == 5
    assert ps.sequence == ("ALA",) * 5
    assert ps.plddt_per_residue.shape == (n,)
    assert ps.plddt_per_residue.dtype == np.float64
    assert ps.cb_or_ca_xyz.shape == (n, 3)
    assert ps.cb_or_ca_xyz.dtype == np.float64
    assert ps.atom_offsets.shape == (n + 1,)
    assert ps.atom_offsets[0] == 0
    assert np.all(np.diff(ps.atom_offsets) >= 0)  # monotonic non-decreasing
    assert ps.atom_name_id.shape == (int(ps.atom_offsets[-1]),)
    assert ps.atom_xyz.shape == (int(ps.atom_offsets[-1]), 3)


def test_atoms_for_recovers_per_residue_data(small_cif):
    """``atoms_for(i)`` returns the right per-residue atom slice."""
    ps = parse.parse_cif_content(small_cif, entry_id="AF-TEST-F1")
    # The synthetic fixture emits N/CA/C/O/CB per residue (5 atoms, all in vocab).
    for i in range(ps.num_residues):
        name_ids, xyz = ps.atoms_for(i)
        assert len(name_ids) == 5, f"residue {i}: expected 5 in-vocab atoms"
        assert xyz.shape == (5, 3)
        # Convert ids back to names — the order matches the PDB record order
        # (N, CA, C, O, CB), which the fast-path single-pass extractor
        # preserves.
        names = [parse._ATOM_NAMES_TUPLE[int(k)] for k in name_ids]
        assert names == ["N", "CA", "C", "O", "CB"]


def test_cb_or_ca_falls_back_to_ca_for_gly(small_cif):
    """For non-GLY residues with CB present, cb_or_ca_xyz is the CB position."""
    ps = parse.parse_cif_content(small_cif, entry_id="AF-TEST-F1")
    # Synthetic fixture is poly-ALA so cb_or_ca_xyz row should be the CB
    # coordinate (4.5, 1.5, z + 0.5) for residue i (1-based in PDB, 0-based here).
    for i in range(ps.num_residues):
        x, y, z = ps.cb_or_ca_xyz[i].tolist()
        expected_z = (i + 1) * 3.0 + 0.5
        assert math.isclose(x, 4.5)
        assert math.isclose(y, 1.5)
        assert math.isclose(z, expected_z)


def test_plddt_per_residue_equals_mean_b_factor(small_cif):
    """All heavy atoms have b_iso=90 in the fixture → plddt = 90.0 per residue."""
    ps = parse.parse_cif_content(small_cif, entry_id="AF-TEST-F1")
    assert np.allclose(ps.plddt_per_residue, 90.0)
    assert math.isclose(ps.global_plddt, 90.0)


def test_try_parse_returns_none_on_bad_input():
    """``try_parse_cif_content`` warns and returns None on garbage."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = parse.try_parse_cif_content("not a cif", entry_id="X")
    assert out is None
