# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The associative (parallel) scan must equal the textbook sequential recurrence.

This is the load-bearing correctness claim for the XLA port: ``mamba.py`` replaced
the sequential ``for t in range(L)`` selective scan with a Hillis-Steele
associative scan for a compact XLA graph. The other tests only compare
padded-vs-unpadded (both using the *parallel* scan), so this one independently
pins parallel == sequential — including the edge cases (L=1, L not a power of
two) where an off-by-one in the scan would hide. No network / checkpoint needed.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mamba import MambaMixer  # noqa: E402


def _sequential_scan(u, dt, A, Bmat, Cmat):
    """Reference: the plain sequential recurrence h_t = dA_t*h_{t-1} + dBu_t."""
    bsz, length, d_inner = u.shape
    dA = torch.exp(dt.unsqueeze(-1) * A)
    dBu = dt.unsqueeze(-1) * Bmat.unsqueeze(2) * u.unsqueeze(-1)
    h = torch.zeros(bsz, d_inner, A.shape[1])
    ys = []
    for t in range(length):
        h = dA[:, t] * h + dBu[:, t]
        ys.append(torch.einsum("bds,bs->bd", h, Cmat[:, t]))
    return torch.stack(ys, dim=1)


def test_parallel_scan_matches_sequential():
    torch.manual_seed(0)
    # (B, L, d_inner, d_state) — include L=1 and non-power-of-two lengths.
    for bsz, length, d_inner, d_state in [(1, 1, 4, 3), (2, 7, 8, 4), (1, 513, 16, 8)]:
        u = torch.randn(bsz, length, d_inner)
        dt = F.softplus(torch.randn(bsz, length, d_inner))
        A = -torch.exp(torch.randn(d_inner, d_state))
        Bmat = torch.randn(bsz, length, d_state)
        Cmat = torch.randn(bsz, length, d_state)
        want = _sequential_scan(u, dt, A, Bmat, Cmat)
        got = MambaMixer._selective_scan(u, dt, A, Bmat, Cmat)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-4), f"mismatch at {(bsz, length, d_inner, d_state)}"
