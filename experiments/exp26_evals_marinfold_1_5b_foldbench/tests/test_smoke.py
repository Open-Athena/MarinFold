# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cheap smoke tests for exp20.

These don't run the model — they just verify that the bin scheme,
CSV schema, and CIF parser line up the way the scoring code assumes.
Run via ``uv run pytest tests/``.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "exp1_document_structures_contacts_and_distances_v1"))

from canonical_sequence import read_canonical_sequence, representative_atom_name
from score_comparison import merge_and_summarize
from score_marinfold import (
    MARINFOLD_BINS, _INRANGE_MIN_A, _INRANGE_MAX_A, _CONTACT_CUTOFF_A,
)


def test_marinfold_bin_midpoints():
    """0.5 Å bins → midpoints 0.25, 0.75, ..., 31.75 Å."""
    mp = MARINFOLD_BINS.midpoints_A
    assert mp.shape == (64,)
    assert mp[0] == pytest.approx(0.25)
    assert mp[-1] == pytest.approx(31.75)
    assert mp[1] - mp[0] == pytest.approx(0.5)


def test_contact_bin_mask_first_16_bins():
    """≤ 8 Å contact mask = first 16 bins (centers 0.25..7.75 Å)."""
    mask = MARINFOLD_BINS.contact_bin_mask
    # bin 15 has center 7.75 (included); bin 16 has center 8.25 (excluded).
    assert mask[15]
    assert not mask[16]
    assert int(mask.sum()) == 16


def test_inrange_filter_intersects_with_protenix():
    """The in-range pair filter uses the Protenix narrower bounds."""
    # 2.3125 Å is Protenix's min_bin from exp12; 21.6875 Å is max_bin.
    # These must match exactly so the cross-model comparison is fair.
    assert _INRANGE_MIN_A == 2.3125
    assert _INRANGE_MAX_A == 21.6875
    assert _CONTACT_CUTOFF_A == 8.0


def _protenix_dir_or_skip() -> Path:
    p = _HERE / "protenix_data" / "data" / "protenix-foldbench-monomers"
    if not p.exists() or not list((p / "gt").glob("*.cif")):
        pytest.skip("protenix_data not present; run fetch_protenix_data.py first")
    return p


def test_canonical_sequence_5sbj():
    """Smallest FoldBench monomer: 30-residue 5sbj_A.

    Sequence parses to 30 residues. 5sbj is a designed peptide and
    has 4 non-canonical residues in entity_poly_seq (mapped to UNK by
    our reader); the canonical 26 should be standard L-AAs.
    """
    protenix_dir = _protenix_dir_or_skip()
    seq = read_canonical_sequence(protenix_dir / "gt" / "5sbj_A.cif")
    assert seq.n_residues == 30, f"expected 30, got {seq.n_residues}"
    n_unk = sum(1 for r in seq.residue_names if r == "UNK")
    assert n_unk == 4, f"expected 4 UNK in 5sbj_A; got {n_unk}: {seq.residue_names!r}"


def test_representative_atom_convention():
    """GLY and UNK both use CA; standard residues use CB."""
    assert representative_atom_name("GLY") == "CA"
    assert representative_atom_name("UNK") == "CA"
    assert representative_atom_name("ALA") == "CB"


def test_score_comparison_writes_clean_summary_and_json_verdict():
    """Summary stays parseable as CSV; verdict details live in JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        marinfold_csv = tmp / "marinfold_scores.csv"
        protenix_csv = tmp / "protenix_scores.csv"
        summary_csv = tmp / "scores_summary.csv"
        scores_csv = tmp / "scores.csv"
        verdict_json = tmp / "hypothesis_verdict.json"

        marinfold_csv.write_text(
            "pdb_id,chain_id,method,n_residues,mae_distogram_cb_angstrom,drmsd_distogram_cb_angstrom,"
            "n_mae_distogram_pairs,mae_distogram_cb_contact_angstrom,drmsd_distogram_cb_contact_angstrom,"
            "n_mae_distogram_contact_pairs,prec_short_L,prec_short_L_2,prec_short_L_5,prec_medium_L,"
            "prec_medium_L_2,prec_medium_L_5,prec_long_L,prec_long_L_2,prec_long_L_5,n_short_contacts,"
            "n_medium_contacts,n_long_contacts,lddt_distogram_cb,lddt_distogram_cb_soft\n"
            "1abc,A,marinfold_1b,10,5.0,6.0,1,5.0,6.0,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,1,1,0.3,0.4\n"
        )
        protenix_csv.write_text(
            "pdb_id,chain_id,mode,n_residues,mae_distogram_cb_angstrom,drmsd_distogram_cb_angstrom,"
            "n_mae_distogram_pairs,mae_distogram_cb_contact_angstrom,drmsd_distogram_cb_contact_angstrom,"
            "n_mae_distogram_contact_pairs,prec_short_L,prec_short_L_2,prec_short_L_5,prec_medium_L,"
            "prec_medium_L_2,prec_medium_L_5,prec_long_L,prec_long_L_2,prec_long_L_5,n_short_contacts,"
            "n_medium_contacts,n_long_contacts,lddt_distogram_cb,lddt_distogram_cb_soft\n"
            "1abc,A,single_seq,10,4.0,5.0,1,4.0,5.0,1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,1,1,0.2,0.3\n"
            "1abc,A,msa,10,1.0,2.0,1,1.0,2.0,1,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1,1,1,0.9,0.95\n"
        )

        verdict = merge_and_summarize(
            marinfold_csv=marinfold_csv,
            protenix_csv=protenix_csv,
            out_scores_csv=scores_csv,
            out_summary_csv=summary_csv,
            out_verdict_json=verdict_json,
        )

        with summary_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["method"] == "marinfold_1b"
        assert "not_supported" not in summary_csv.read_text()

        verdict_data = json.loads(verdict_json.read_text())
        assert verdict_data["verdict"] == verdict["verdict"] == "not_supported"
        assert any("marinfold_1b=0.3000" in detail for detail in verdict_data["details"])


def test_smoke_outputs_if_present():
    """If a local smoke run has happened, sanity-check the distogram .npz.

    Only runs if ``outputs/5sbj_A/distogram.npz`` exists — i.e. after
    ``uv run python run_1b_eval.py --limit 1``.
    """
    npz = _HERE / "outputs" / "5sbj_A" / "distogram.npz"
    if not npz.exists():
        pytest.skip("no smoke outputs yet")
    with np.load(npz) as data:
        probs = data["probs"]
    assert probs.shape == (30, 30, 64), probs.shape
    # Distogram is symmetric: probs[i, j, :] == probs[j, i, :].
    np.testing.assert_array_equal(probs, probs.transpose(1, 0, 2))
    # Diagonal is zeros (we skip self-pairs).
    assert np.all(probs[np.arange(30), np.arange(30), :] == 0)
    # Off-diagonal rows sum to ~1 (renormalized prob distribution).
    sums = probs.sum(axis=-1)
    iu = np.triu_indices(30, k=1)
    off_diag_sums = sums[iu]
    assert np.allclose(off_diag_sums, 1.0, atol=1e-3), (
        f"off-diagonal pair sums should ~1: min={off_diag_sums.min()}, "
        f"max={off_diag_sums.max()}"
    )
