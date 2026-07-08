# Copyright The MarinFold Authors
# Copyright John (Zhiyao) Ma (@johnma2006)
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch, XLA-safe Mamba-1 encoder/decoder for bio2token.

Why this exists: bio2token's published encoder runs ``mamba_ssm``'s custom
CUDA kernels (``selective_scan_cuda`` / ``causal_conv1d``), which do not
exist on TPU/XLA. This module reimplements the same Mamba-1 (S6) block in
plain PyTorch ops so the forward pass compiles under ``torch_xla`` and also
runs on CPU/MPS for local development.

The inner mixer is structurally the well-known reference
`johnma2006/mamba-minimal` (https://github.com/johnma2006/mamba-minimal) —
identical canonical layer names (``in_proj / conv1d / x_proj / dt_proj /
A_log / D / out_proj``) and the same sequential selective scan. What we add
on top is bio2token's stack semantics: a **bidirectional, weight-shared**
pass (run each block forward and on the flipped sequence, then sum), an
input/output projection, and a final ``norm_f`` — matching the parameter
layout of the official checkpoint exactly (verified: state_dict loads with
no missing/unexpected keys). See ``model.py`` for the checkpoint loader.

Perf note for the XLA port: ``_selective_scan`` is a sequential recurrence
over the length axis. It is correct on every backend but, unrolled, builds
a large XLA graph; replacing it with an associative/parallel scan (and
static-length bucketing) is the throughput work for TPU. Correctness does
not depend on that change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Weight-only RMSNorm (matches mamba_ssm's ``RMSNorm``; eps 1e-5)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class MambaMixer(nn.Module):
    """Mamba-1 (S6) mixer. Checkpoint shapes: d_model=128, d_inner=256,
    d_state=16, dt_rank=8, d_conv=4."""

    def __init__(self, d_model=128, d_inner=256, d_state=16, dt_rank=8, d_conv=4):
        super().__init__()
        self.d_inner, self.d_state, self.dt_rank, self.d_conv = d_inner, d_state, dt_rank, d_conv
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                groups=d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):  # x: (B, L, d_model)
        L = x.shape[1]
        xz = self.in_proj(x)
        xin, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)
        xc = self.conv1d(xin.transpose(1, 2))[..., :L]   # depthwise causal conv
        xc = F.silu(xc.transpose(1, 2))
        dt, Bmat, Cmat = torch.split(
            self.x_proj(xc), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())               # (d_inner, d_state)
        y = self._selective_scan(xc, dt, A, Bmat, Cmat)
        y = y + xc * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    @staticmethod
    def _selective_scan(u, dt, A, Bmat, Cmat):
        # u, dt: (B, L, d_inner); A: (d_inner, d_state); B, C: (B, L, d_state)
        Bsz, L, d_inner = u.shape
        dA = torch.exp(dt.unsqueeze(-1) * A)                          # (B, L, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * Bmat.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, d_state)
        h = torch.zeros(Bsz, d_inner, A.shape[1], device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]
            ys.append(torch.einsum("bds,bs->bd", h, Cmat[:, t]))
        return torch.stack(ys, dim=1)                                 # (B, L, d_inner)


class Block(nn.Module):
    """Prenorm block (mamba_ssm style): out = mixer(norm(hidden + residual))."""

    def __init__(self, d_model=128):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model=d_model)

    def forward(self, hidden_states, residual):
        residual = hidden_states if residual is None else (hidden_states + residual)
        hidden_states = self.norm(residual)
        return self.mixer(hidden_states), residual


class MambaStack(nn.Module):
    """bio2token's bidirectional, weight-shared Mamba stack.

    Mirrors ``bio2token.layers.mamba.MambaStack``: input projection, N
    prenorm blocks each run forward AND on the flipped sequence (shared
    weights, outputs summed), a final ``norm_f``, and an output projection.
    """

    def __init__(self, d_input, d_output, d_model=128, n_layer=4, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_projection = (nn.Identity() if d_input == d_model
                                 else nn.Linear(d_input, d_model, bias=False))
        self.layers = nn.ModuleList([Block(d_model) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.output_projection = (nn.Identity() if d_output == d_model
                                  else nn.Linear(d_model, d_output, bias=False))

    def forward(self, input_ids):  # (B, L, d_input)
        hidden_states = self.input_projection(input_ids)
        residual = None
        for layer in self.layers:
            h_fwd, r_fwd = layer(hidden_states, residual)
            if self.bidirectional:
                h_bwd, r_bwd = layer(hidden_states.flip(1),
                                     None if residual is None else residual.flip(1))
                hidden_states = h_fwd + h_bwd.flip(1)
                residual = r_fwd + r_bwd.flip(1)
            else:
                hidden_states, residual = h_fwd, r_fwd
        residual = hidden_states if residual is None else (hidden_states + residual)
        return self.output_projection(self.norm_f(residual))
