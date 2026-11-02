# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The PyTorch smear and NoPE modules, with no transformers dependency.

Kept separate from ``arms.py`` so ``gpu/tests/test_cross_implementation.py`` can
import it alongside the levanter implementation: the two projects pin different
transformers majors, but both can hold plain torch modules.
"""

import torch
import torch.nn.functional as functional
from torch import nn

# Matches ``gpu/architecture.GATE_DIMS``: the gate reads a slice of the token
# embedding, as in the speedrun's smear module.
GATE_DIMS = 16


class SmearEmbedding(nn.Module):
    """Token embedding plus a gated, per-channel causal smear.

    ``x_t = e_t + sum_k sigmoid(gate_k(e_t)) * (w_k * e_{t-k})``

    ``w_k`` are separate ``hidden``-shaped vectors per offset — a shared scalar
    would make offsets 1 and 2 indistinguishable and collapse the arg1/arg2
    distinction inside a ``<contact> <pX> <pY>`` statement. They start at zero,
    so a smear arm and its control are the same function at step 0.
    """

    def __init__(self, table: nn.Embedding, width: int):
        super().__init__()
        self.table = table
        self.width = width
        hidden = table.embedding_dim
        self.gate_dims = min(GATE_DIMS, hidden)
        self.gate = nn.Linear(self.gate_dims, width)
        self.weights = nn.Parameter(torch.zeros(width, hidden))
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.table(input_ids)
        gates = torch.sigmoid(self.gate(embeddings[..., : self.gate_dims]))
        total = embeddings
        for offset in range(1, self.width + 1):
            # Pad at the FRONT and drop the tail: a shift, never a roll. A roll
            # would wrap the last tokens of the sequence into the first
            # positions, which leaks the future into the past.
            shifted = functional.pad(embeddings[:, :-offset], (0, 0, offset, 0))
            total = total + gates[..., offset - 1 : offset] * (self.weights[offset - 1] * shifted)
        return total


class NoRotaryEmbedding(nn.Module):
    """Rotary drop-in that applies no rotation: cos = 1, sin = 0."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        cos, sin = self.inner(x, position_ids)
        return torch.ones_like(cos), torch.zeros_like(sin)


__all__ = ["GATE_DIMS", "NoRotaryEmbedding", "SmearEmbedding"]
