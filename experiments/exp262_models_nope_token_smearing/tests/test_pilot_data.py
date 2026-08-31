# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The pilot's packing contract: attention and loss stop at document boundaries."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot"))

from train_pilot import document_mask  # noqa: E402

EOS = 1


def test_document_mask_blocks_across_boundaries():
    """Three packed documents; nothing may look back past its own <eos>."""
    ids = torch.tensor([[5, 6, EOS, 7, 8, 9, EOS, 3, 4]])
    attention, loss_mask = document_mask(ids, EOS)

    assert attention[0, 0, 2].int().tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0], (
        "the <eos> belongs to the document it ends, not the one that follows"
    )
    assert attention[0, 0, 3].int().tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0], (
        "the first token of a document may see only itself"
    )
    assert attention[0, 0, 5].int().tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert attention[0, 0, 8].int().tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_loss_mask_drops_the_impossible_targets():
    """Predicting a document's first token from the document before it is noise."""
    ids = torch.tensor([[5, 6, EOS, 7, 8, 9, EOS, 3, 4]])
    _attention, loss_mask = document_mask(ids, EOS)
    assert loss_mask[0].int().tolist() == [1, 1, 0, 1, 1, 1, 0, 1, 1]


def test_mask_is_causal_within_a_document():
    ids = torch.tensor([[5, 6, 7, 8]])
    attention, loss_mask = document_mask(ids, EOS)
    expected = torch.ones(4, 4, dtype=torch.bool).tril()
    assert torch.equal(attention[0, 0], expected)
    assert loss_mask.all(), "a single document loses no targets"
