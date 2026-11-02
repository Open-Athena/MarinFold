# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-B tests. The GPU ones need the `mpnn` extra and a CUDA device."""

from __future__ import annotations

import sys
from pathlib import Path

import gemmi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone import (  # noqa: E402
    backbone_coords,
    prepare_structure,
    residue_sequence,
    strip_to_backbone,
)
from redesign import (  # noqa: E402
    DESIGN_TEMPERATURES,
    BackboneEntry,
    batch_by_exact_length,
    design_batch,
)

PDB_MIRROR = Path("/data/tim/af3-db/mmcif_files")


def _entry(stem: str) -> BackboneEntry:
    path = PDB_MIRROR / f"{stem}.cif"
    if not path.exists():
        pytest.skip(f"{path} not in the local PDB mirror")
    bb = strip_to_backbone(prepare_structure(gemmi.read_structure(str(path))))
    _chains, coords = backbone_coords(bb)
    return BackboneEntry(stem, residue_sequence(bb), coords)


def _devices() -> list[str]:
    """CPU always (it is the production device — see generate_rows), CUDA too
    when there is a usable one."""
    torch = pytest.importorskip("torch")
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


DEVICES = _devices() if __import__("importlib").util.find_spec("torch") else []


def test_batches_are_single_length() -> None:
    entries = [
        BackboneEntry(f"e{i}", "A" * length, [])
        for i, length in enumerate([100, 100, 100, 250, 250, 800])
    ]
    batches = batch_by_exact_length(
        entries, max_batch=2, max_batch_residues=10**9, designs_per_backbone=1
    )
    assert all(len({e.length for e in b}) == 1 for b in batches)
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == len(entries)


def test_residue_bound_accounts_for_design_replication() -> None:
    entries = [BackboneEntry(f"e{i}", "A" * 100, []) for i in range(16)]
    # 100 residues x 8 designs = 800 padded residues per backbone.
    batches = batch_by_exact_length(
        entries, max_batch=64, max_batch_residues=1600, designs_per_backbone=8
    )
    assert max(len(b) for b in batches) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_length_batch_is_rejected(device: str) -> None:
    with pytest.raises(ValueError, match="one length per batch"):
        design_batch([_entry("1crn"), _entry("1ubq")], device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_per_item_temperature_matches_scalar(device: str) -> None:
    """A `[B, 1]` temperature tensor == the scalar it replicates.

    design_batch folds the 8 design slots into the batch dimension and passes
    a per-item temperature, relying on ProteinMPNN using `temperature` only in
    broadcast-compatible divisions. If that ever stops holding, every sequence
    we generate is drawn at the wrong temperature — silently. So: run the same
    batch with a uniform ladder and assert it reproduces the scalar path.
    """
    torch = pytest.importorskip("torch")
    from proteinmpnn.protein_mpnn_utils import tied_featurize

    from redesign import _batch_entry, _batch_seed, MPNN_ALPHABET, load_model
    import numpy as np

    entries = [_entry("1crn")] * 4
    model = load_model(device)
    feats = tied_featurize([_batch_entry(e) for e in entries], device, None)
    (X, S_true, mask, _l, chain_M, chain_enc, *_rest) = feats
    chain_M_pos, omit_AA_mask, residue_idx = feats[10], feats[11], feats[12]
    pssm_coef, pssm_bias, pssm_log_odds, bias_by_res = feats[15:19]

    omit = np.zeros(len(MPNN_ALPHABET), dtype=np.float32)
    omit[MPNN_ALPHABET.index("X")] = 1.0
    bias = np.zeros(len(MPNN_ALPHABET), dtype=np.float32)

    def sample(temperature):
        with torch.no_grad():
            torch.manual_seed(_batch_seed(entries))
            randn = torch.randn(chain_M.shape, device=device)
            return model.sample(
                X, randn, S_true, chain_M, chain_enc, residue_idx, mask=mask,
                temperature=temperature, omit_AAs_np=omit, bias_AAs_np=bias,
                chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
                pssm_log_odds_flag=False, pssm_log_odds_mask=pssm_log_odds,
                pssm_bias_flag=False, bias_by_res=bias_by_res,
            )["S"]

    scalar = sample(0.2)
    tensor = sample(torch.full((len(entries), 1), 0.2, device=device))
    assert torch.equal(scalar, tensor)


@pytest.mark.parametrize("device", DEVICES)
def test_design_batch_shape_and_content(device: str) -> None:
    """Also the numpy-2 canary: `proteinmpnn` pins `numpy<2`, we override it
    (see pyproject), and this is the test that actually runs the model under
    the overridden version."""
    entry = _entry("1crn")
    designs = design_batch([entry, entry], device=device)

    assert len(designs) == 2 * len(DESIGN_TEMPERATURES)
    assert [d.design_index for d in designs[: len(DESIGN_TEMPERATURES)]] == list(
        range(len(DESIGN_TEMPERATURES))
    )
    for d in designs:
        assert len(d.sequence) == entry.length
        assert "X" not in d.sequence          # omit_AAs must be respected
        assert 0.0 <= d.identity_to_native <= 1.0
        assert d.mpnn_score > 0.0

    # Low temperature recovers more of the native sequence than high.
    low = [d.identity_to_native for d in designs if d.mpnn_temperature == 0.1]
    high = [d.identity_to_native for d in designs if d.mpnn_temperature == 0.5]
    assert sum(low) / len(low) > sum(high) / len(high)
