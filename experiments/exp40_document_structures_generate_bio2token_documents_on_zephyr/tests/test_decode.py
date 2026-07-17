# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Document-level round-trip: a ``bio2token-v2`` document decodes to a structure.

Two properties:

1. **Lossless invertibility** — building a document (which shuffles the atom
   triples) and parsing it back recovers the *exact* per-atom codes in the
   encoder's canonical order. The self-describing format loses no information.
2. **CA-RMSD within the reconstruction floor** — decoding a document to
   coordinates reproduces the original CA positions to sub-1.5 Å. bio2token is a
   lossy (FSQ-quantized) autoencoder, so this floor is *not* zero; it is the
   ceiling on how well any model trained on these tokens could reconstruct
   structure, which is exactly why it's worth pinning.

Marked ``network``: downloads the pretrained checkpoint on first run.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIF = os.path.join(HERE, "tests", "data", "1QYS.cif")
sys.path.insert(0, HERE)

import torch  # noqa: E402

from adapt import parse_structure, to_bio2token_batch  # noqa: E402
from decode import canonical_indices, decode_document, parse_document  # noqa: E402
from generate import generate_document, get_model  # noqa: E402


def _kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d])) @ U.T
    return torch.sqrt(((R @ Pc.T).T - Qc).pow(2).sum(1).mean()).item()


@pytest.mark.network
def test_document_is_losslessly_invertible():
    model = get_model("cpu")
    batch = to_bio2token_batch(parse_structure(CIF), add_batch_dim=False)
    original = model.tokenize(batch["structure"][None])[0].tolist()

    doc = generate_document(CIF)["document"]
    sequence, atoms = parse_document(doc)
    recovered, _ = canonical_indices(sequence, atoms)

    # The per-document shuffle is undone exactly: same codes, same canonical order.
    assert recovered == original


@pytest.mark.network
def test_document_ca_rmsd_within_reconstruction_floor():
    model = get_model("cpu")
    parsed = parse_structure(CIF)
    batch = to_bio2token_batch(parsed, add_batch_dim=False)

    coords, order = decode_document(generate_document(CIF)["document"], model)
    ca = [i for i, (_, atom) in enumerate(order) if atom == "CA"]
    assert len(ca) == len(parsed.residues)  # one CA per residue

    ca_rmsd = _kabsch_rmsd(coords[ca].float(), batch["structure"][ca].float())
    assert ca_rmsd < 1.5, f"CA-RMSD {ca_rmsd:.3f} Å exceeds bio2token's floor"
