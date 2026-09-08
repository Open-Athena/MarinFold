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


def test_section_mask_handles_windows_that_open_mid_document():
    """Evaluation windows start at arbitrary offsets, so section state cannot be counted.

    The earlier implementation compared cumulative counts of <begin_statements>
    and <contacts-v1>. On a window opening mid-document those counters stay
    equal through every later sequence section, so all of them were scored as
    structure — biasing the section split the pilot's mechanistic claim rested on.
    """
    import torch

    from train_pilot import section_mask

    ids = {"doc_type": 2, "begin_statements": 9, "end": 10}
    # ...sequence | <begin_statements> structure | <contacts-v1> sequence | <begin_statements> structure
    window = torch.tensor([[7, 7, 9, 5, 5, 2, 7, 7, 9, 5, 5]])
    structure, known = section_mask(window, ids)

    assert known[0].tolist() == [False, False, True, True, True, True, True, True, True, True, True]
    assert structure[0].tolist() == [
        False, False, True, True, True, False, False, False, True, True, True
    ], "the second document's sequence section must not be scored as structure"


def test_section_mask_is_correct_on_a_clean_boundary():
    import torch

    from train_pilot import section_mask

    ids = {"doc_type": 2, "begin_statements": 9, "end": 10}
    window = torch.tensor([[2, 7, 7, 9, 5, 5, 2, 7, 9, 5]])
    structure, known = section_mask(window, ids)
    assert known.all()
    assert structure[0].tolist() == [False, False, False, True, True, True, False, False, True, True]


def test_loss_mask_covers_a_window_ending_on_eos():
    """The last position's target lives outside the window and can still be impossible.

    Comparing neighbouring segment ids leaves the final entry at its initialised
    True, so a window ending on <eos> trained one cross-document target.
    """
    import torch

    from train_pilot import document_mask

    ids = torch.tensor([[5, 6, EOS, 7, EOS]])
    _attention, loss_mask = document_mask(ids, EOS)
    assert loss_mask[0].tolist() == [True, True, False, True, False]
