# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness check for the MLX backend's prefix-cache path.

The MLX backend runs the shared prompt prefix forward once into a
``KVCache``, then replicates that cache across a batch of tails and
runs a single forward over the tails. This must produce numerically
identical probabilities to the naive baseline of concatenating each
``(prefix + tail)`` and running a full forward — modulo bf16
rounding.

Marked ``network``: downloads the 1B model from HF if it's not in the
HF cache. Run with::

    uv run pytest tests/test_mlx_prefix_cache.py -v
"""

import math

import numpy as np
import pytest


pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402

from marinfold import load_backend  # noqa: E402
from marinfold.inference._mlx import MlxBackend  # noqa: E402


def _full_forward_probs(
    model,
    prefix: list[int],
    tail: list[int],
    target_ids: list[int],
) -> np.ndarray:
    """Reference: full ``(prefix + tail)`` forward, no cache reuse."""
    full = mx.array([list(prefix) + list(tail)], dtype=mx.int32)
    logits = model(full)[:, -1, :]
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    targets = mx.array(target_ids, dtype=mx.int32)
    return np.asarray(probs[:, targets], dtype=np.float32)[0]


@pytest.mark.network
def test_mlx_prefix_cache_matches_full_recompute() -> None:
    backend = load_backend("mlx", model="1B", tail_batch_size=8)
    assert isinstance(backend, MlxBackend)
    model = backend._model  # noqa: SLF001 — test cares about internals

    # Vocabulary is small (~2840 tokens); pick a few real token ids for
    # the prefix and tails so we exercise embedding lookups that
    # actually trigger the model's RoPE / attention paths rather than
    # whatever lives at id 0/1.
    tok = backend.tokenizer
    prefix = tok.encode(
        "<contacts-and-distances-v1> <begin_sequence> "
        "<ALA> <GLY> <VAL> <LEU> <ILE> <PRO> <PHE> <TRP> <MET> <SER> "
        "<begin_statements>",
        add_special_tokens=False,
    )
    tails = [
        tok.encode(f"<distance> <p1> <p{j}> <CA> <CA>", add_special_tokens=False)
        for j in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)  # 10 tails, force chunking
    ]
    # Sanity: every prompt encoded 1:1.
    assert all(len(t) == 5 for t in tails), [len(t) for t in tails]
    targets = tok.encode(
        " ".join(f"<d{(k + 1) * 0.5:.1f}>" for k in range(64)),
        add_special_tokens=False,
    )
    assert len(targets) == 64

    cached = backend.next_token_probs(prefix, tails, targets)
    assert cached.shape == (len(tails), len(targets))

    uncached = np.stack(
        [_full_forward_probs(model, prefix, tail, targets) for tail in tails]
    )

    # bf16 weights → ~1e-3 absolute / ~1e-2 relative. The two paths
    # do the same arithmetic in different orders, so small drift is
    # expected. Probabilities are bounded in [0, 1], so atol is the
    # right metric.
    np.testing.assert_allclose(cached, uncached, atol=1e-3, rtol=0)

    # Sanity: cached probs are real probabilities.
    assert np.all(cached >= 0.0)
    assert np.all(cached <= 1.0)
    assert math.isfinite(cached.sum())
