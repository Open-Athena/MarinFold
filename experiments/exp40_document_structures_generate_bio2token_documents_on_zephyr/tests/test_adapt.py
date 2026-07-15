# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the production adapter + document generator.

The adapter (gemmi, full precision) must reproduce bio2token's own input
pipeline (the biopython/pandas oracle in ``reference_input.py``). Two
independent checks: the atom *layout* must match exactly (``token_class``
identical — a precision-free check of canonical ordering + class assignment),
and the resulting *tokens* must agree and reconstruct (a numeric check).

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
from generate import generate_document  # noqa: E402
from model import load_bio2token  # noqa: E402
from reference_input import build_reference_batch, kabsch_rmsd  # noqa: E402
from vocab import BEGIN_SEQUENCE, BEGIN_STATEMENTS, END, STRUCTURE_TOKEN  # noqa: E402


@pytest.mark.network
def test_adapter_matches_oracle():
    parsed = parse_structure(CIF)
    batch = to_bio2token_batch(parsed)
    ref = build_reference_batch(CIF)

    # Same atom count and canonical layout (precision-independent).
    assert batch["structure"].shape == ref["structure"].shape
    assert torch.equal(batch["token_class"], ref["token_class"])
    # residue_index/atom_name provenance covers every kept atom.
    assert batch["residue_index"].shape[1] == batch["structure"].shape[1]
    assert len(batch["atom_name"]) == batch["structure"].shape[1]

    model = load_bio2token()
    ours = model.tokenize(batch["structure"])[0]
    theirs = model.tokenize(ref["structure"])[0]
    agree = (ours == theirs).float().mean().item()
    assert agree >= 0.98, f"token agreement {agree:.3f} too low vs oracle"

    recon, _ = model.reconstruct(batch["structure"])
    assert kabsch_rmsd(recon[0].float(), batch["structure"][0].float()) < 1.5


@pytest.mark.network
def test_generate_document_wellformed():
    row = generate_document(CIF)
    parsed = parse_structure(CIF)
    n_res, n_atom = len(parsed.residues), batch_atoms(parsed)

    doc = row["document"]
    assert doc.startswith(f"{STRUCTURE_TOKEN} {BEGIN_SEQUENCE} ")
    assert f" {BEGIN_STATEMENTS} " in doc
    assert doc.endswith(f" {END}")
    # one code per atom; num_tokens = markers(4) + sequence pairs (2/residue)
    # + atom triples (3/atom).
    assert doc.count("<bt") == row["num_atoms"] == n_atom
    assert row["seq_length"] == n_res
    assert row["num_tokens"] == 4 + 2 * n_res + 3 * n_atom
    # 1QYS fits the default context budget, so no atoms are sampled out.
    assert row["truncated"] is False
    assert row["num_atoms_total"] == n_atom


def batch_atoms(parsed) -> int:
    return to_bio2token_batch(parsed)["structure"].shape[1]
