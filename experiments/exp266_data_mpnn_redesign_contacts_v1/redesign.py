# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B worker — ProteinMPNN redesign of one batch of backbones.

No zephyr import, so it is unit-testable and runnable on a workstation GPU
before it ever reaches CoreWeave. ``cli.py`` wraps ``design_batch`` in a
``map_shard``.

Design decisions worth knowing:

* **Batched, bucketed by length.** ``tied_featurize`` pads a batch to its
  longest member, so batching length-sorted backbones keeps padding (and
  therefore wasted FLOPs) small. Per-row threading — the right answer for
  the I/O-bound contacts-v1 pipelines — is the wrong shape here: accelerator
  throughput comes from batching, not concurrent single-item forwards. See
  the ``zephyr-pipeline-performance`` skill's neural-tokenizer section.
* **``augment_eps=0.0``.** ``v_48_020`` was *trained* with 0.20 Å backbone
  noise; adding noise at inference would design against perturbed geometry
  that no longer matches the coordinates Stage C computes contacts from.
  ``protein_mpnn_run.py`` defaults to the same 0.00.
* **The published sequence table is the reproducibility artifact.**
  ProteinMPNN's sampler draws from the global torch RNG inside
  ``model.sample``, so bit-reproducing a design would mean reproducing the
  exact batch composition. Rather than monkey-patch the sampler we seed per
  batch and *publish every sampled sequence*: Stage C is then fully
  deterministic given Stage B's output, which is what the corpus needs.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# ProteinMPNN's fixed alphabet; index 20 is 'X'.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# 8 designs per backbone on a near-native -> diverse temperature ladder.
# Regeneration is the expensive step, so the axis is baked in once and a
# downstream training experiment can subset by `mpnn_temperature` without
# re-running anything.
DESIGN_TEMPERATURES: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5)

DEFAULT_WEIGHTS = "v_48_020"


@dataclass(frozen=True)
class BackboneEntry:
    """One monomer backbone ready for ProteinMPNN."""

    entry_id: str
    native_sequence: str
    # [L, 4, 3] in N, CA, C, O order — `backbone.backbone_coords` output.
    coords: Any

    @property
    def length(self) -> int:
        return len(self.native_sequence)


@dataclass(frozen=True)
class Design:
    entry_id: str
    design_index: int
    sequence: str
    mpnn_temperature: float
    mpnn_score: float          # mean per-residue NLL of the sampled sequence
    identity_to_native: float  # fraction of positions equal to the native AA


@functools.cache
def load_model(device: str = "cuda", weights: str = DEFAULT_WEIGHTS):
    """Load ProteinMPNN once per process (module-level memoization).

    A Zephyr/Iris worker serves many shards; re-loading the checkpoint per
    shard is the 41-hour-cluster-bill mistake the pipeline skill calls out.
    """
    import torch
    from proteinmpnn import protein_mpnn_utils
    from proteinmpnn.data import vanilla_model_weights

    path = f"{vanilla_model_weights.__path__[0]}/{weights}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = protein_mpnn_utils.ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,          # deterministic geometry; see module docstring
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def _batch_entry(entry: BackboneEntry) -> dict[str, Any]:
    """The dict shape ``tied_featurize`` expects, for a single chain 'A'.

    Built directly rather than by round-tripping through ProteinMPNN's
    ``parse_PDB``: serializing 4 M backbones to PDB text and re-parsing them
    with a pure-Python parser would be pure dead time on a GPU worker.
    """
    coords = np.asarray(entry.coords, dtype=np.float32)
    return {
        "name": entry.entry_id,
        "num_of_chains": 1,
        "seq": entry.native_sequence,
        "seq_chain_A": entry.native_sequence,
        "coords_chain_A": {
            "N_chain_A": coords[:, 0].tolist(),
            "CA_chain_A": coords[:, 1].tolist(),
            "C_chain_A": coords[:, 2].tolist(),
            "O_chain_A": coords[:, 3].tolist(),
        },
    }


def _batch_seed(entries: list[BackboneEntry]) -> int:
    """A stable seed from batch composition, so a re-run of the *same* batch
    reproduces the same designs."""
    h = hashlib.sha256("|".join(e.entry_id for e in entries).encode())
    return int.from_bytes(h.digest()[:8], "big") % (2**63)


def design_batch(
    entries: list[BackboneEntry],
    *,
    device: str = "cuda",
    weights: str = DEFAULT_WEIGHTS,
    temperatures: tuple[float, ...] = DESIGN_TEMPERATURES,
) -> list[Design]:
    """Sample ``len(temperatures)`` sequences for every backbone, in ONE pass.

    Every entry must have the same length (see :func:`batch_by_exact_length`).

    The batch is replicated ``len(temperatures)`` times and the per-item
    temperature is passed as a ``[B*T, 1]`` tensor rather than a scalar.
    ``sample()`` only ever uses ``temperature`` in broadcast-compatible
    divisions (``W_out(h_V_t) / temperature`` on a ``[B, 21]`` logit tensor),
    so this is a supported tensor argument, not a patch —
    ``tests/test_redesign.py::test_per_item_temperature_matches_scalar`` pins
    the equivalence.

    It matters because ProteinMPNN's decode is ``L`` *sequential* steps whose
    cost barely moves with batch size: measured on an RTX A5000 at L=154, 8
    designs took 10.4 s at B=1 and 16.6 s at B=64 (1298 -> 32 ms/sequence).
    Folding the 8 design slots into the batch dimension buys most of another
    8x on top.
    """
    import torch
    from proteinmpnn.protein_mpnn_utils import _scores, tied_featurize

    if not entries:
        return []
    lengths = {e.length for e in entries}
    if len(lengths) != 1:
        raise ValueError(
            f"design_batch requires one length per batch, got {sorted(lengths)}; "
            "use batch_by_exact_length()"
        )

    model = load_model(device, weights)
    n_designs = len(temperatures)
    # Replicate entry-major: [e0 x T, e1 x T, ...] so a protein's designs stay
    # adjacent and the output order is deterministic.
    replicated = [e for e in entries for _ in temperatures]
    batch = [_batch_entry(e) for e in replicated]

    (X, S_true, mask, _lengths, chain_M, chain_encoding_all, _letters, _visible,
     _masked, _chain_lengths, chain_M_pos, omit_AA_mask, residue_idx,
     _dihedral, _tied, pssm_coef, pssm_bias, pssm_log_odds,
     bias_by_res, _tied_beta) = tied_featurize(batch, device, None)

    # Never sample 'X': contacts-v1 serializes it as <UNK> and it has no
    # rotamer, so it would silently drop every contact it participates in.
    omit_AAs_np = np.zeros(len(MPNN_ALPHABET), dtype=np.float32)
    omit_AAs_np[MPNN_ALPHABET.index("X")] = 1.0
    bias_AAs_np = np.zeros(len(MPNN_ALPHABET), dtype=np.float32)

    temperature = torch.tensor(
        [t for _ in entries for t in temperatures], device=device
    ).view(-1, 1)

    with torch.no_grad():
        torch.manual_seed(_batch_seed(entries))
        randn = torch.randn(chain_M.shape, device=device)
        out = model.sample(
            X, randn, S_true, chain_M, chain_encoding_all, residue_idx,
            mask=mask, temperature=temperature,
            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
            chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
            pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
            pssm_log_odds_flag=False, pssm_log_odds_mask=pssm_log_odds,
            pssm_bias_flag=False, bias_by_res=bias_by_res,
        )
        S_sample = out["S"]
        # Teacher-force the sampled sequence to score it (mean per-residue
        # NLL) — the same quantity protein_mpnn_run.py reports.
        log_probs = model(
            X, S_sample, mask, chain_M * chain_M_pos, residue_idx,
            chain_encoding_all, randn,
            use_input_decoding_order=True, decoding_order=out["decoding_order"],
        )
        scores = _scores(S_sample, log_probs, mask * chain_M * chain_M_pos)

    S_np = S_sample.cpu().numpy()
    scores_np = scores.cpu().numpy()
    designs: list[Design] = []
    for row, entry in enumerate(replicated):
        design_index = row % n_designs
        sequence = "".join(MPNN_ALPHABET[t] for t in S_np[row, : entry.length])
        identity = sum(
            a == b for a, b in zip(sequence, entry.native_sequence)
        ) / entry.length
        designs.append(
            Design(
                entry_id=entry.entry_id,
                design_index=design_index,
                sequence=sequence,
                mpnn_temperature=temperatures[design_index],
                mpnn_score=float(scores_np[row]),
                identity_to_native=identity,
            )
        )
    return designs


def batch_by_exact_length(
    entries: list[BackboneEntry],
    *,
    max_batch: int,
    max_batch_residues: int,
    designs_per_backbone: int = len(DESIGN_TEMPERATURES),
) -> list[list[BackboneEntry]]:
    """Group backbones into batches in which **every member has the same length**.

    Not an optimization — a correctness requirement. ProteinMPNN's
    ``tied_featurize`` pads ``omit_AA_mask`` (an ``[L, 21]`` array) with the
    1-D pad spec ``[[0, L_max - l]]``, which widens the *alphabet* axis
    instead of the length axis and raises on any batch whose members differ
    in length::

        ValueError: could not broadcast input array from shape (476,426)
                    into shape (476,21)

    Upstream never hits it because the stock CLI batches N copies of a single
    protein. Rather than monkey-patch a dependency (root ``AGENTS.md``), we
    feed it only the batch shape it handles.

    We can afford this: at 4 M backbones over ~800 distinct lengths the
    equal-length groups are large, and the batches carry *zero* padding, so
    exact-length batching is also the cheapest possible forward pass. Stage A
    sorts the manifest by ``seq_len`` so each shard covers a narrow length
    band and these groups stay full.
    """
    by_length: dict[int, list[BackboneEntry]] = {}
    for entry in entries:
        by_length.setdefault(entry.length, []).append(entry)

    batches: list[list[BackboneEntry]] = []
    for length, group in sorted(by_length.items()):
        # Bound by count and by total residues. The residue bound is what
        # actually protects device memory (64 x 120 residues and 64 x 1,900
        # are very different forward passes), and it must account for the
        # `designs_per_backbone` replication design_batch applies.
        per_backbone = max(length * designs_per_backbone, 1)
        cap = max(1, min(max_batch, max_batch_residues // per_backbone))
        for start in range(0, len(group), cap):
            batches.append(group[start : start + cap])
    return batches
