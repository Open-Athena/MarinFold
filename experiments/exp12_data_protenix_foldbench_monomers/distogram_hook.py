# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Protenix's distogram-head outputs via ``register_forward_hook``.

Standard PyTorch idiom — no monkey-patching, no vendor fork.

Protenix's inference path (bytedance/Protenix protenix/model/protenix.py)
computes ``self.distogram_head(z)`` once per trunk forward (so once per
seed) and immediately collapses the resulting ``[N_token, N_token, 64]``
logits to scalar contact probabilities. We hook the distogram_head
module to grab the raw logits before that happens, softmax along the
bin axis, and save them to disk.

Usage::

    from distogram_hook import DistogramCapture
    cap = DistogramCapture(out_dir=Path("/scratch/out/single_seq/5sbj_A"))
    handle = cap.attach(runner.model.distogram_head)
    try:
        cap.set_current_seed(seed)
        infer_predict(runner, configs)
    finally:
        handle.remove()

Each call to the hook writes one ``..._distogram.npz`` (with key
``probs`` of shape ``[N, N, 64]``, float32) to::

    out_dir/seed_<seed>/<stem>_distogram.npz

where ``<stem>`` matches Protenix's sample_name. The current seed is
set explicitly by the caller (see the seed loop in
``runner.inference.infer_predict``).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class DistogramCapture:
    """Forward-hook callable that writes per-(seed, stem) distogram ``.npz``s.

    ``out_dir`` is the root of the per-(mode, stem) output (e.g.
    ``/scratch/out/single_seq/5sbj_A``). Per-seed subdirectories are
    created as needed. Stem and seed are set externally before each
    seed's forward pass; the hook does not try to infer them.
    """

    out_dir: Path
    current_stem: str | None = None
    current_seed: int | None = None
    n_writes: int = 0

    def set_current(self, stem: str, seed: int) -> None:
        self.current_stem = stem
        self.current_seed = seed

    def __call__(self, module: Any, inputs: Any, output: Any) -> None:  # noqa: ARG002
        # output: torch.Tensor of shape [..., N, N, 64] (logits).
        # We softmax along the last axis and dump probabilities as float32.
        if self.current_stem is None or self.current_seed is None:
            logger.warning(
                "DistogramCapture hook fired with no current (stem, seed) set — "
                "skipping (this shouldn't happen)."
            )
            return
        import torch
        if not isinstance(output, torch.Tensor):
            logger.warning(
                "DistogramCapture: expected output tensor, got %s; skipping.",
                type(output).__name__,
            )
            return
        # Last dim should be 64 bins. Earlier dims should be [..., N, N].
        # Squeeze any leading batch dim so the saved shape is exactly [N, N, 64].
        logits = output.detach().float()
        while logits.ndim > 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        if logits.ndim != 3 or logits.shape[-1] != 64:
            logger.warning(
                "DistogramCapture: unexpected output shape %s; skipping.",
                tuple(logits.shape),
            )
            return
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        seed_dir = self.out_dir / f"seed_{self.current_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        path = seed_dir / f"{self.current_stem}_distogram.npz"
        np.savez_compressed(path, probs=probs)
        self.n_writes += 1
        logger.info(
            "DistogramCapture: wrote %s (shape=%s, write #%d)",
            path, probs.shape, self.n_writes,
        )

    def attach(self, distogram_head_module: Any):
        """Register on the given module; returns the handle (caller removes)."""
        return distogram_head_module.register_forward_hook(self)
