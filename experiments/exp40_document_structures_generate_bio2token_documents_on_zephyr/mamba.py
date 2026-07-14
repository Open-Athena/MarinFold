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
A_log / D / out_proj``) and the same sequential selective scan. Added on top
is bio2token's stack semantics: a **bidirectional, weight-shared** pass (run
each block forward and on the flipped sequence, then sum), an input/output
projection, and a final ``norm_f`` — matching the parameter layout of the
official checkpoint exactly, so its ``state_dict`` loads with no missing or
unexpected keys. See ``model.py`` for the checkpoint loader.

XLA port: ``_selective_scan`` is an associative (Hillis-Steele) scan over the
length axis — log2(L) fully-vectorized steps, no Python loop — so it compiles
to a compact, fast XLA graph instead of an L-deep unrolled recurrence. It is
numerically identical to the textbook sequential recurrence, pinned to float
precision by ``tests/test_scan.py`` (incl. L=1 and non-power-of-two lengths).
Combined with static-length bucketing in the tokenizer (pad each structure to
one of a few fixed lengths so XLA compiles once per bucket), this is what makes
TPU inference practical.

Bucketing pads sequences on the right, which requires a ``mask`` (1 for real
atoms, 0 for padding): the stack is **bidirectional**, so end-padding would
otherwise leak into real positions through the backward (flipped) scan and the
conv1d bias. Masking zeroes the hidden state and the scan input at padded
positions, making the padded forward pass bit-for-bit equal to the unpadded
one on the real positions (validated in tests/test_bucketing.py).
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

    def forward(self, x, mask=None):  # x: (B, L, d_model); mask: (B, L) or None
        L = x.shape[1]
        xz = self.in_proj(x)
        xin, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)
        xc = self.conv1d(xin.transpose(1, 2))[..., :L]   # depthwise causal conv
        xc = F.silu(xc.transpose(1, 2))
        if mask is not None:
            # Zero the padded positions' scan input so ``dBu`` there is 0 and the
            # recurrence carries no state through padding (load-bearing for the
            # backward pass; see module docstring). The conv1d bias otherwise
            # makes padding non-zero even for all-zero inputs.
            xc = xc * mask.unsqueeze(-1)
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
        """Associative (Hillis-Steele) inclusive scan of the linear recurrence
        ``h_t = dA_t * h_{t-1} + dBu_t``, then ``y_t = <C_t, h_t>``.

        Each step composes the per-position affine maps ``x -> a*x + b`` using
        the associative operator ``(a2,b2) ∘ (a1,b1) = (a2*a1, a2*b1 + b2)``.
        log2(L) vectorized steps, no Python loop over L — XLA compiles this to
        a compact graph. Identical to the sequential recurrence to float eps.
        """
        # u, dt: (B, L, d_inner); A: (d_inner, d_state); B, C: (B, L, d_state)
        L = u.shape[1]
        a = torch.exp(dt.unsqueeze(-1) * A)                          # (B, L, d_inner, d_state)
        b = dt.unsqueeze(-1) * Bmat.unsqueeze(2) * u.unsqueeze(-1)   # (B, L, d_inner, d_state)
        shift = 1
        while shift < L:
            # prev[t] = scan value at t-shift; identity (a=1, b=0) for t < shift.
            a_prev = F.pad(a[:, :L - shift], (0, 0, 0, 0, shift, 0), value=1.0)
            b_prev = F.pad(b[:, :L - shift], (0, 0, 0, 0, shift, 0), value=0.0)
            b = a * b_prev + b        # compose (outer=current, inner=prev); uses old a
            a = a * a_prev
            shift *= 2
        return torch.einsum("blds,bls->bld", b, Cmat)                # (B, L, d_inner)


class Block(nn.Module):
    """Prenorm block (mamba_ssm style): out = mixer(norm(hidden + residual))."""

    def __init__(self, d_model=128):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model=d_model)

    def forward(self, hidden_states, residual, mask=None):
        residual = hidden_states if residual is None else (hidden_states + residual)
        hidden_states = self.norm(residual)
        return self.mixer(hidden_states, mask), residual


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

    def forward(self, input_ids, mask=None):  # (B, L, d_input); mask: (B, L) or None
        hidden_states = self.input_projection(input_ids)
        residual = None
        # Flipped mask for the backward pass; zeroing padded hidden states each
        # layer keeps the conv1d from reading non-zero padding as a neighbor.
        mask_c = None if mask is None else mask.unsqueeze(-1).to(hidden_states.dtype)
        mask_flip = None if mask is None else mask.flip(1)
        for layer in self.layers:
            if mask_c is not None:
                hidden_states = hidden_states * mask_c
                residual = residual if residual is None else residual * mask_c
            h_fwd, r_fwd = layer(hidden_states, residual, mask)
            if self.bidirectional:
                h_bwd, r_bwd = layer(hidden_states.flip(1),
                                     None if residual is None else residual.flip(1),
                                     mask_flip)
                hidden_states = h_fwd + h_bwd.flip(1)
                residual = r_fwd + r_bwd.flip(1)
            else:
                hidden_states, residual = h_fwd, r_fwd
        residual = hidden_states if residual is None else (hidden_states + residual)
        return self.output_projection(self.norm_f(residual))
