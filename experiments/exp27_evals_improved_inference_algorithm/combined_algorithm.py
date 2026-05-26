# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Library function: the headline inference algorithm from exp27.

The algorithm is ``iter_R4_grow_on_sampled_uniform_M5``:

  Stage A — sampled-uniform contact prior:
    Sample M=5 independent contact-only rollouts from the base prompt
    using a custom ``logits_processor`` that masks all tokens except
    the 3 ``<*-range-contact>`` tokens and the N position tokens. The
    range-token sub-distribution is overwritten to uniform to undo the
    1B model's ~99% medium-range prior. Take the union of unique
    (i, j) pairs across rollouts as the seed prefix; do one distance
    readout under that prefix.

  Stage B — iterative growing-K refinement:
    For 4 rounds, pick top-K contacts from the previous round's
    distogram (range-ordered long>medium>short, ``min_contact_prob=0.1``),
    re-read distances under the new prefix. K per round grows
    0.5L, 1.0L, 1.5L, 2.5L.

Standard expected-distance readout (``E[d] = sum(probs * midpoints)``)
on the final distogram. No post-hoc sharpening (sharpening only helps
when context is bad).

Usage:

  from naive_inference import load_runtime
  from combined_algorithm import predict_distogram_combined
  from canonical_sequence import read_canonical_sequence
  from gt_filtered_inference import build_gt_shell_mask
  from pathlib import Path

  rt = load_runtime(model_nickname="1B", models_yaml=Path("..."))
  seq = read_canonical_sequence(gt_cif_path)
  pair_mask = build_gt_shell_mask(gt_cif_path, seq.n_residues)
  probs = predict_distogram_combined(
      rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
  )
  # probs is (N, N, 64) — pass to score_marinfold.score_one via a tmp .npz.

The default hyperparameters reproduce the +42.81%-on-train headline.
**They are tuned for the 1B checkpoint.** Per RESULTS_LOG.md, the
algorithm gives +31.75% on a held-out protein set but only +9.04% on
the 1.5B model — knob re-tuning is needed for different model
checkpoints.

This module is library-only: no CLI, no file I/O in the core
function. The script `run_iterative.py` (using
``--prior-name distogram_sampled_uniform_M5_union.npz``) reproduces
the same algorithm via the existing two-stage CLI when on-disk
caching of intermediate distograms is desired.
"""

import math
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from naive_inference import Runtime  # noqa: E402

# We re-use the heavy lifters from the per-stage modules so that the
# library function is byte-equivalent to running the experiment scripts.
# Any algorithm bug fixed here must be fixed there too.
from sampled_contacts_inference import (  # noqa: E402
    sample_contact_prefix,
    predict_distogram_with_prefix,
)
from iterative_inference import (  # noqa: E402
    extract_round_statements,
    predict_distogram_with_statements,
)


# Default knobs. These are the values that produced the +42.81% headline
# on the 1B train set. Treat them as the "approved" defaults; pass
# overrides via keyword arguments if you want to explore.
DEFAULTS = {
    # Stage A (sampled-uniform prior)
    "n_rollouts": 5,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_sample_tokens": 900,
    "range_strategy": "uniform",
    "base_seed": 27,
    # Stage B (iterative growing-K)
    "n_rounds": 4,
    "k_contacts_per_L_per_round": (0.5, 1.0, 1.5, 2.5),
    "min_contact_prob": 0.1,
    "order": "long_med_short",
    "min_modal_prob": 0.5,  # only used if k_distances > 0 (we don't, by default)
    "sharpen_T_for_modes": 0.1,  # idem
    # Readout
    "batch_size": 128,
    "top_k_logprobs": 128,
}


def predict_distogram_combined(
    *,
    rt: Runtime,
    residue_names: list[str],
    pair_mask: np.ndarray,
    n_rollouts: int = DEFAULTS["n_rollouts"],
    temperature: float = DEFAULTS["temperature"],
    top_p: float = DEFAULTS["top_p"],
    max_sample_tokens: int = DEFAULTS["max_sample_tokens"],
    range_strategy: str = DEFAULTS["range_strategy"],
    base_seed: int = DEFAULTS["base_seed"],
    n_rounds: int = DEFAULTS["n_rounds"],
    k_contacts_per_L_per_round: tuple[float, ...] = DEFAULTS["k_contacts_per_L_per_round"],
    min_contact_prob: float = DEFAULTS["min_contact_prob"],
    order: str = DEFAULTS["order"],
    batch_size: int = DEFAULTS["batch_size"],
    include_history: bool = False,
    include_intermediate_distograms: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run the headline exp27 algorithm on one protein.

    Args:
        rt: A loaded :class:`naive_inference.Runtime` (one vLLM
            instance pinned to the current GPU).
        residue_names: Canonical 3-letter residue names, length N.
            Get this from
            ``canonical_sequence.read_canonical_sequence(gt_cif).residue_names``.
        pair_mask: ``[N, N]`` bool, ``True`` for the (i, j) upper-
            triangle pairs to read distances at. Typically the
            LDDT-shell mask from
            ``gt_filtered_inference.build_gt_shell_mask(gt_cif, N)``.
            Pairs outside the mask are not queried and stay as
            zero rows in the returned probs.
        n_rollouts, temperature, top_p, max_sample_tokens,
        range_strategy, base_seed: Stage A knobs. Defaults reproduce
            the +42.81% train-set headline. ``range_strategy=uniform``
            is the fix for the 1B model's ~99% medium-range prior;
            other models may want a different strategy
            (see ``probe_range_entropy.py``).
        n_rounds, k_contacts_per_L_per_round, min_contact_prob, order:
            Stage B knobs.
        batch_size: vLLM generate() batch size for the distance
            readouts.
        include_history: If True, ``meta`` includes the FULL contact
            picks made at every step — each Stage A rollout's emitted
            ``(i, j, range_token)`` triples, the union seed list, and
            each Stage B round's chosen contact statements. Useful for
            debugging which seeds the algorithm actually committed to,
            or for re-running just one stage. Default False (keeps the
            ``meta`` dict small).
        include_intermediate_distograms: If True, ``meta`` also
            includes the Stage A distogram and each round's post-round
            distogram (each is ``[N, N, 64]`` float32 — these are big).
            Useful for cross-stage comparison; default False.

    Returns:
        ``(probs, meta)``:
          - ``probs``: ``[N, N, 64]`` float32 distogram. Symmetric
            across the diagonal. Pairs outside ``pair_mask`` are zero.
            Score with
            ``score_marinfold.score_one(distogram_npz=...path to a saved npz of these probs...)``.
          - ``meta``: dict with the algorithm name, knobs used, and
            per-stage counts (seeds, pairs queried, prefix token
            count). With ``include_history=True`` also contains the
            actual contact picks at every step; with
            ``include_intermediate_distograms=True`` also contains the
            intermediate distograms.
    """
    N = len(residue_names)
    if pair_mask.shape != (N, N):
        raise ValueError(
            f"pair_mask shape {pair_mask.shape} != ({N}, {N}) (residue_names)"
        )

    rollouts_meta: list[dict] = []

    # ---------------- Stage A: sampled-uniform union prior ----------------
    range_log_weights = None  # only used by range_strategy='weighted'
    all_seeds: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(n_rollouts):
        seeds, n_sample_tokens = sample_contact_prefix(
            rt, residue_names,
            max_tokens=max_sample_tokens,
            temperature=temperature, top_p=top_p,
            seed=base_seed + r,
            range_strategy=range_strategy,
            range_log_weights=range_log_weights,
        )
        rollout_record = {
            "rollout": r, "seed": base_seed + r,
            "n_sample_tokens": n_sample_tokens, "n_seeds": len(seeds),
        }
        if include_history:
            # Each entry is (i, j, range_token_string); 1-indexed positions.
            rollout_record["seeds"] = list(seeds)
        rollouts_meta.append(rollout_record)
        for (i, j, tok) in seeds:
            if (i, j) in seen:
                continue
            seen.add((i, j))
            all_seeds.append((i, j, tok))

    # Sort the union (long > medium > short, then by (i, j)) to match the
    # production prefix order used by the experiment scripts.
    range_idx = {
        "<long-range-contact>": 0,
        "<medium-range-contact>": 1,
        "<short-range-contact>": 2,
    }
    all_seeds.sort(key=lambda s: (range_idx[s[2]], s[0], s[1]))

    stage_a_probs, n_pairs_queried_a, prefix_tok_a = predict_distogram_with_prefix(
        rt=rt, residue_names=residue_names, pair_mask=pair_mask,
        seeds=all_seeds, batch_size=batch_size,
    )

    # ---------------- Stage B: iterative growing-K refinement ----------------
    current_prior = stage_a_probs
    rounds_meta: list[dict] = []
    intermediate_round_distograms: list[np.ndarray] = []
    already_committed: set[tuple[int, int]] = set()
    for round_idx in range(n_rounds):
        k_contacts = int(round(k_contacts_per_L_per_round[round_idx] * N))
        contact_statements, distance_statements = extract_round_statements(
            current_prior,
            k_contacts=k_contacts,
            k_distances=0,  # default schedule has no distance commits
            min_contact_prob=min_contact_prob,
            min_modal_prob=DEFAULTS["min_modal_prob"],
            sharpen_T_for_modes=DEFAULTS["sharpen_T_for_modes"],
            already_committed=already_committed,
            residue_names=residue_names,
            order=order,
        )
        current_prior, n_pairs_queried_b, prefix_tok_b = predict_distogram_with_statements(
            rt=rt, residue_names=residue_names, pair_mask=pair_mask,
            contact_statements=contact_statements,
            distance_statements=distance_statements,
            batch_size=batch_size,
        )
        for (i, j, *_r) in distance_statements:
            already_committed.add((i, j))
        round_record = {
            "round": round_idx,
            "k_contacts": k_contacts,
            "n_contact_statements": len(contact_statements),
            "n_distance_statements": len(distance_statements),
            "n_pairs_queried": n_pairs_queried_b,
            "prefix_token_count": prefix_tok_b,
        }
        if include_history:
            # contact_statements: [(i, j, "<{range}-range-contact>")]
            # distance_statements: [(i, j, "<atom_i>", "<atom_j>", "<d_X.X>")]
            round_record["contact_statements"] = list(contact_statements)
            round_record["distance_statements"] = list(distance_statements)
        if include_intermediate_distograms:
            intermediate_round_distograms.append(current_prior.copy())
        rounds_meta.append(round_record)

    stage_a_meta = {
        "n_rollouts": n_rollouts,
        "temperature": temperature,
        "top_p": top_p,
        "max_sample_tokens": max_sample_tokens,
        "range_strategy": range_strategy,
        "base_seed": base_seed,
        "rollouts": rollouts_meta,
        "n_union_seeds": len(all_seeds),
        "n_pairs_queried": n_pairs_queried_a,
        "prefix_token_count": prefix_tok_a,
    }
    if include_history:
        stage_a_meta["union_seeds"] = list(all_seeds)
    if include_intermediate_distograms:
        stage_a_meta["distogram"] = stage_a_probs

    stage_b_meta = {
        "n_rounds": n_rounds,
        "k_contacts_per_L_per_round": list(k_contacts_per_L_per_round),
        "min_contact_prob": min_contact_prob,
        "order": order,
        "rounds": rounds_meta,
    }
    if include_intermediate_distograms:
        stage_b_meta["round_distograms"] = intermediate_round_distograms

    meta = {
        "algorithm": "iter_R4_grow_on_sampled_uniform_M5",
        "n_residues": N,
        "stage_a": stage_a_meta,
        "stage_b": stage_b_meta,
    }
    return current_prior, meta


def predict_distogram_combined_from_cif(
    *,
    rt: Runtime,
    gt_cif: Path,
    **algorithm_kwargs,
) -> tuple[np.ndarray, dict]:
    """Convenience wrapper: derive ``residue_names`` and the LDDT-shell
    pair mask from a Protenix GT CIF, then call
    :func:`predict_distogram_combined`.

    The LDDT-shell mask is what makes the readout cheap (~7x speedup
    on the train set) while keeping ``lddt_distogram_cb`` identical
    by construction. For a non-GT-aware run on held-out proteins,
    pass ``pair_mask = np.triu(np.ones((N, N), dtype=bool), k=1)``
    to :func:`predict_distogram_combined` directly.
    """
    from canonical_sequence import read_canonical_sequence
    from gt_filtered_inference import build_gt_shell_mask
    seq = read_canonical_sequence(gt_cif)
    pair_mask = build_gt_shell_mask(gt_cif, seq.n_residues)
    return predict_distogram_combined(
        rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
        **algorithm_kwargs,
    )
