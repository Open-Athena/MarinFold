# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness oracle for the pure-PyTorch bio2token reimplementation.

The single load-bearing assertion is reconstruction RMSD: a protein can only
round-trip through encoder -> FSQ -> decoder to sub-Angstrom accuracy if BOTH
the official weights loaded correctly AND the pure-PyTorch selective scan is
numerically faithful to the original CUDA Mamba. A wrong scan yields garbage
coordinates, not a near-miss. Keep this assertion green after any change to
``mamba.py``.

Marked ``network``: downloads the ~14 MB pretrained checkpoint on first run.
Run from the experiment dir: ``uv run pytest tests/ -m network``.
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIF = os.path.join(HERE, "tests", "data", "1QYS.cif")

# The modules under test live in the experiment dir as plain modules (repo
# convention: run tests with the experiment dir on sys.path).
sys.path.insert(0, HERE)

from model import CODEBOOK_SIZE, load_bio2token  # noqa: E402
from reference_input import build_reference_batch, kabsch_rmsd  # noqa: E402


@pytest.mark.network
def test_1qys_roundtrip_and_tokens():
    model = load_bio2token()  # downloads checkpoint if absent; strict load
    batch = build_reference_batch(CIF)
    structure = batch["structure"]  # (1, L, 3)
    L = structure.shape[1]

    recon, indices = model.reconstruct(structure)
    tokens = indices[0]

    # One token per atom, all within the FSQ codebook.
    assert tokens.shape[0] == L
    assert int(tokens.min()) >= 0 and int(tokens.max()) < CODEBOOK_SIZE

    # The gate: faithful reconstruction (weights + selective scan both correct).
    rmsd = kabsch_rmsd(recon[0].float(), structure[0].float())
    assert rmsd < 1.5, f"reconstruction RMSD {rmsd:.3f} A too high — impl diverges"


@pytest.mark.network
def test_device_agnostic_tokens_match_cpu():
    """Tokens are identical on MPS/CUDA if available (device-agnostic forward)."""
    structure = build_reference_batch(CIF)["structure"]
    cpu_tokens = load_bio2token(device="cpu").tokenize(structure).tolist()

    alt = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else None)
    if alt is None:
        pytest.skip("no non-CPU device available")
    alt_tokens = load_bio2token(device=alt).tokenize(structure.to(alt)).cpu().tolist()
    assert alt_tokens == cpu_tokens
