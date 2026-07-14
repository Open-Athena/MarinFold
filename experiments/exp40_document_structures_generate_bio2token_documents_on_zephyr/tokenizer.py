# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Accelerator tokenizer backend: bucketed, batched, device-agnostic.

The bio2token encoder is compute-bound (a Mamba scan, ~0.19 ms/atom on CPU),
which inverts the usual document-pipeline regime where per-row GCS I/O
dominates. Two facts drive this module:

1. **XLA compiles once per input shape.** Feeding the raw variable length of
   every structure would trigger a fresh (multi-second) compile per length.
   The fix is to **bucket**: pad each structure up to the smallest fixed length
   in a short geometric ladder, so a whole run only ever compiles
   ~``len(BUCKETS)`` graphs. Padding is masked (see ``mamba.py``) so real-atom
   tokens are unchanged.

2. **Accelerators want batches.** A single structure (B=1) starves a TPU's
   systolic array. Structures are grouped by bucket and each group runs as one
   batched forward pass, which is where the throughput comes from.

The public surface is :func:`tokenize_structures` (list of per-structure coord
tensors -> list of per-structure token tensors) plus the device helpers
:func:`resolve_device` / :func:`sync_device`. It is deliberately free of any
Zephyr / marin import so it unit-tests locally on CPU.
"""

from collections import defaultdict

import torch

# Atom-count buckets (geometric, ~2x). A structure of ``n`` atoms is padded up
# to the smallest bucket >= n. The ladder tops out at 16384 atoms (~2000
# residues), which covers typical protein inputs; a larger structure falls back
# to its own exact length (one extra XLA compile, rare) rather than being
# dropped or truncated.
DEFAULT_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384)
# Upper bound on the structure *count* in one batched forward pass.
DEFAULT_MAX_BATCH = 32
# Upper bound on *padded tokens* (B * bucket_len) in one batched forward pass —
# the real HBM governor. The associative scan holds ~4 live (B, L, d_inner,
# d_state) buffers ≈ B*L*64 KiB, so a fixed structure count is catastrophic for
# the largest bucket: 32 * 16384 * 64 KiB ≈ 34 GB exceeds a v6e chip's ~31 GB
# HBM and out-of-memories. Budgeting by B*L makes big buckets take small batches
# and small buckets big ones, keeping peak HBM flat. 131072 -> ~8.6 GB scan
# peak, ample headroom.
DEFAULT_MAX_BATCH_TOKENS = 131072


def resolve_device(name: str):
    """Return a torch device for ``name``. ``"xla"`` initializes torch_xla.

    On a Zephyr TPU worker, Iris has already set ``PJRT_DEVICE=TPU``, so
    ``torch_xla.device()`` binds to the local chip with no extra env setup.
    """
    if name == "xla":
        import torch_xla
        return torch_xla.device()
    return torch.device(name)


def sync_device(device) -> None:
    """Flush the lazy XLA graph (no-op on non-XLA devices).

    Called once per batched forward pass so the XLA graph is materialized and
    doesn't grow unbounded across a shard's many batches.
    """
    if getattr(device, "type", None) == "xla":
        import torch_xla
        torch_xla.sync()


def bucket_for(n: int, buckets: tuple[int, ...] = DEFAULT_BUCKETS) -> int:
    """Smallest bucket length >= ``n``; if ``n`` exceeds the ladder, ``n`` itself.

    Returning ``n`` for oversized structures keeps them correct at the cost of
    one extra (rare) XLA compile, rather than silently dropping or truncating.
    """
    for b in buckets:
        if n <= b:
            return b
    return n


def batch_size_for(bucket_len: int, max_batch: int, max_batch_tokens: int) -> int:
    """Structures per forward pass for a bucket: the smaller of the count cap and
    the token budget ``max_batch_tokens // bucket_len``, but always >= 1 (a lone
    oversized structure still runs)."""
    return max(1, min(max_batch, max_batch_tokens // bucket_len))


def _iter_batches(indices: list[int], max_batch: int):
    for i in range(0, len(indices), max_batch):
        yield indices[i:i + max_batch]


@torch.no_grad()
def tokenize_structures(
    structures: list[torch.Tensor],
    model,
    *,
    device=None,
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    max_batch: int = DEFAULT_MAX_BATCH,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
) -> list[torch.Tensor]:
    """Tokenize a list of per-structure coord tensors ``(L_i, 3)``.

    Groups structures by length bucket, runs each group as batched, right-padded
    + masked forward passes on ``device``, and returns a list of 1-D token
    tensors (on CPU) aligned to the input order, each sliced back to its true
    length. An empty ``structures`` returns ``[]``.

    Each bucket's batch size is ``max(1, min(max_batch, max_batch_tokens //
    bucket_len))`` — the token budget keeps peak device memory flat across
    buckets (see ``DEFAULT_MAX_BATCH_TOKENS``); a single structure always runs
    even if it alone exceeds the budget.
    """
    if not structures:
        return []
    device = device if device is not None else next(model.parameters()).device

    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(structures):
        groups[bucket_for(int(s.shape[0]), buckets)].append(i)

    results: list[torch.Tensor | None] = [None] * len(structures)
    for blen, idxs in groups.items():
        batch_size = batch_size_for(blen, max_batch, max_batch_tokens)
        for chunk in _iter_batches(idxs, batch_size):
            coords = torch.zeros(len(chunk), blen, 3, dtype=torch.float32)
            mask = torch.zeros(len(chunk), blen, dtype=torch.float32)
            lengths = []
            for j, i in enumerate(chunk):
                s = structures[i]
                n = int(s.shape[0])
                coords[j, :n] = s
                mask[j, :n] = 1.0
                lengths.append(n)
            tokens = model.tokenize(coords.to(device), mask.to(device))
            sync_device(device)
            tokens = tokens.to("cpu")
            for j, i in enumerate(chunk):
                results[i] = tokens[j, :lengths[j]].clone()
    return results  # type: ignore[return-value]
