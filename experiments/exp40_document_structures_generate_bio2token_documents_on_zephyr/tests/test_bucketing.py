# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Bucketing + masking must not change the tokens of real atoms.

The XLA path pads every structure up to a fixed bucket length and runs it
batched. Because the Mamba stack is bidirectional, end-padding could leak into
real positions through the backward scan and the conv1d bias — the mask in
``mamba.py`` is what prevents that. These tests pin the contract that the
padded, batched tokenization is *bit-for-bit* equal to the unpadded one on the
real atoms, so bucketing is a pure throughput optimization with no effect on
the corpus.

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
from model import load_bio2token  # noqa: E402
from tokenizer import batch_size_for, bucket_for, tokenize_structures  # noqa: E402


@pytest.mark.network
def test_bucketed_batch_matches_unpadded():
    model = load_bio2token(device="cpu")
    coords = to_bio2token_batch(parse_structure(CIF), add_batch_dim=False)["structure"]
    n = coords.shape[0]

    # Reference: unpadded, no mask.
    ref = model.tokenize(coords[None])[0]

    # Same structure at several truncated lengths, tokenized as one mixed batch.
    # Distinct lengths force multiple buckets and non-trivial padding within a
    # batch; each must reproduce its own unpadded tokenization exactly.
    lengths = [n, n // 2, n // 3, 5]
    structures = [coords[:m] for m in lengths]
    refs = [model.tokenize(coords[:m][None])[0] for m in lengths]

    out = tokenize_structures(structures, model, device=torch.device("cpu"), max_batch=4)
    assert len(out) == len(structures)
    for got, want, m in zip(out, refs, lengths, strict=True):
        assert got.shape == (m,)
        assert torch.equal(got, want), f"length {m}: bucketed tokens differ from unpadded"
    # First entry is the full structure — sanity check it equals the top-level ref.
    assert torch.equal(out[0], ref)


def test_batch_size_respects_token_budget():
    # The token budget (B*L) caps peak HBM: a fixed count OOMs the biggest
    # bucket (32*16384 ≈ 34 GB > v6e HBM). Big buckets must shrink the batch.
    mb, mt = 32, 131072
    for blen in (256, 512, 1024, 2048, 4096, 8192, 16384, 20000):
        bs = batch_size_for(blen, mb, mt)
        assert bs >= 1
        assert bs <= mb
        assert bs * blen <= mt or bs == 1  # within budget (or a lone big struct)
    assert batch_size_for(512, mb, mt) == 32     # small bucket -> count-capped
    assert batch_size_for(16384, mb, mt) == 8    # big bucket -> token-capped
    assert batch_size_for(20000, mb, mt) == 6    # oversized fallback still batches


def test_bucket_for_ladder():
    assert bucket_for(1) == 256
    assert bucket_for(256) == 256
    assert bucket_for(257) == 512
    assert bucket_for(2048) == 2048
    # Oversized structures fall back to their exact length (rare extra compile),
    # never dropped or truncated.
    assert bucket_for(20000) == 20000
