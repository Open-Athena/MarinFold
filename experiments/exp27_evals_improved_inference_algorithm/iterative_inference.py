# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Idea 2: iterative self-distillation.

Round 1 is exactly ``seeded_contacts_inference``: seed K1 contacts
from the baseline_naive prior, re-read distances. Round 2+ uses the
*previous round's* distogram as the prior, extracting both new
high-confidence contacts AND specific committed distances (top of
the sharpened distribution per pair) as statements to prepend.

For each round r >= 1, the prefix is:

  <begin_sequence><AAs><begin_statements>
    <{range}-range-contact><pi><pj>           # K_contacts(r) statements
    ...
    <distance><pi><pj><atom_i><atom_j><d_X.X>  # K_distances(r) statements
    ...

The committed distances are picked from pairs not already covered by
contacts (and not yet committed in previous rounds). For each such
pair, we commit the modal-distance bin (after light sharpening of the
prior distogram, so the commit isn't a regression-to-mean point).

Run R rounds; report each round's LDDT.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from naive_inference import (  # noqa: E402
    BIN_MIDPOINTS,
    DISTANCE_MAX_A,
    NUM_DISTANCE_BINS,
    Runtime,
    _build_base_prompt,
    _encode_tokens,
    load_runtime,
)
from canonical_sequence import (  # noqa: E402
    read_canonical_sequence,
    representative_atom_name,
)
from gt_filtered_inference import build_gt_shell_mask  # noqa: E402
from seeded_contacts_inference import _RANGE_BUCKETS  # noqa: E402

_CONTACT_BIN_COUNT = 16  # 0..8 Å


def _sharpen(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    """Same sharpen as sharpen_sweep.sharpen. Used for picking the modal bin."""
    out = np.zeros_like(probs)
    row_sums = probs.sum(axis=-1)
    nonzero = row_sums > 0
    if not nonzero.any():
        return out
    p = probs[nonzero]
    logp = np.log(p + eps)
    z = logp / T
    z = z - z.max(axis=-1, keepdims=True)
    expz = np.exp(z)
    out[nonzero] = (expz / expz.sum(axis=-1, keepdims=True)).astype(np.float32)
    return out


def _bin_to_distance_token(bin_idx: int) -> str:
    """Map a bin index 0..63 to its ``<d_X.X>`` token string."""
    # vocab.DISTANCE_BINS: [f"<d{i * 0.5:.1f}>" for i in range(1, 65)]
    # so bin 0 corresponds to "<d0.5>" (the upper edge of [0, 0.5))
    return f"<d{(bin_idx + 1) * 0.5:.1f}>"


def extract_round_statements(
    probs: np.ndarray,
    *,
    k_contacts: int,
    k_distances: int,
    min_contact_prob: float,
    min_modal_prob: float,
    sharpen_T_for_modes: float,
    already_committed: set[tuple[int, int]],
    residue_names: list[str],
    order: str = "long_med_short",
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str, str, str]]]:
    """Pick contact + distance statements from a prior distogram.

    ``order`` controls how the contact candidates are sorted before
    truncation to ``k_contacts``:
      - ``long_med_short`` (default): all long-range first, then medium,
        then short, each by descending prob. Matches the v1 vocab order.
      - ``by_prob``: top by prob regardless of range. Tends to give a
        natural mix when high-prob picks span ranges.

    Returns ``(contact_statements, distance_statements)`` where
      contact_statements = [(i, j, "<{range}-range-contact>")]
      distance_statements = [(i, j, "<atom_i>", "<atom_j>", "<d_X.X>")]
    """
    n = probs.shape[0]
    contact_probs = probs[:, :, :_CONTACT_BIN_COUNT].sum(axis=-1)

    contact_candidates: list[tuple[int, int, str, float]] = []
    for _name, sep_lo, sep_hi, token in _RANGE_BUCKETS:
        for i in range(n):
            j_start = i + sep_lo
            j_end = n if sep_hi is None else min(n, i + sep_hi + 1)
            for j in range(j_start, j_end):
                if (i + 1, j + 1) in already_committed:
                    continue
                p = float(contact_probs[i, j])
                if p < min_contact_prob:
                    continue
                contact_candidates.append((i + 1, j + 1, token, p))
    if order == "long_med_short":
        range_index = {tok: idx for idx, (_, _, _, tok) in enumerate(_RANGE_BUCKETS)}
        contact_candidates.sort(key=lambda c: (range_index[c[2]], -c[3]))
    elif order == "by_prob":
        contact_candidates.sort(key=lambda c: -c[3])
    else:
        raise ValueError(f"unknown order: {order}")
    contact_statements = [(i, j, tok) for (i, j, tok, _p) in contact_candidates[:k_contacts]]

    # Track what we've committed to avoid double-counting in distance picks.
    committed_for_round: set[tuple[int, int]] = set(already_committed)
    for (i, j, _tok) in contact_statements:
        committed_for_round.add((i, j))

    # Sharpen for picking the modal bin (sharper = more confident commit).
    sharp = _sharpen(probs, sharpen_T_for_modes)
    modal_bin = sharp.argmax(axis=-1)
    modal_p = sharp.max(axis=-1)

    # Candidate distance pairs: any (i, j) in upper triangle that has data
    # in the prior, isn't already committed, and whose sharpened mass is
    # >= min_modal_prob.
    distance_candidates: list[tuple[int, int, int, float]] = []
    row_sums = probs.sum(axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            if (i + 1, j + 1) in committed_for_round:
                continue
            if row_sums[i, j] <= 0:
                continue
            mp = float(modal_p[i, j])
            if mp < min_modal_prob:
                continue
            distance_candidates.append((i + 1, j + 1, int(modal_bin[i, j]), mp))
    distance_candidates.sort(key=lambda c: -c[3])
    distance_picks = distance_candidates[:k_distances]
    distance_statements: list[tuple[int, int, str, str, str]] = []
    for (i, j, b, _p) in distance_picks:
        a_i = representative_atom_name(residue_names[i - 1])
        a_j = representative_atom_name(residue_names[j - 1])
        distance_statements.append((i, j, f"<{a_i}>", f"<{a_j}>", _bin_to_distance_token(b)))
    return contact_statements, distance_statements


def predict_distogram_with_statements(
    *,
    rt: Runtime,
    residue_names: list[str],
    pair_mask: np.ndarray,
    contact_statements: list[tuple[int, int, str]],
    distance_statements: list[tuple[int, int, str, str, str]],
    batch_size: int = 128,
    top_k_logprobs: int = 128,
) -> tuple[np.ndarray, int, int]:
    """Read distances at pair_mask pairs with full statements prefix."""
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    if pair_mask.shape != (n, n):
        raise ValueError(f"pair_mask shape {pair_mask.shape} != ({n}, {n})")

    base_tokens = _build_base_prompt(residue_names)
    prefix_tokens: list[str] = list(base_tokens)
    for (i, j, tok) in contact_statements:
        prefix_tokens.extend([tok, f"<p{i}>", f"<p{j}>"])
    for (i, j, a_i, a_j, d_tok) in distance_statements:
        prefix_tokens.extend(["<distance>", f"<p{i}>", f"<p{j}>", a_i, a_j, d_tok])
    prefix_ids = _encode_tokens(rt.tokenizer, prefix_tokens)

    distance_id_set = set(rt.distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(rt.distance_token_ids)}

    # Don't re-query pairs we've already committed (their value is
    # already locked by the distance statement). Mask them out of the
    # readout list so we save work AND honour the commitment.
    committed = set()
    for (i, j, *_r) in contact_statements:
        # contact commits a range, not a specific distance — still re-query
        pass
    for (i, j, *_r) in distance_statements:
        committed.add((i, j))

    pairs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if not pair_mask[i - 1, j - 1]:
                continue
            if (i, j) in committed:
                continue
            pairs.append((i, j))

    prompts = []
    for (i, j) in pairs:
        a_i = representative_atom_name(residue_names[i - 1])
        a_j = representative_atom_name(residue_names[j - 1])
        tail = _encode_tokens(rt.tokenizer, [
            "<distance>", f"<p{i}>", f"<p{j}>", f"<{a_i}>", f"<{a_j}>",
        ])
        prompts.append(TokensPrompt(prompt_token_ids=prefix_ids + tail))

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, logprobs=top_k_logprobs, n=1,
    )

    probs = np.zeros((n, n, NUM_DISTANCE_BINS), dtype=np.float32)

    # For pairs that were committed via distance statements, set their
    # row of probs to one-hot on the committed bin. That way scoring
    # treats them as fully-confident at that distance.
    for (i, j, _a_i, _a_j, d_tok) in distance_statements:
        # parse the bin from the token string: "<dX.Y>" -> Y bin index
        # bin k corresponds to "<d{(k+1)*0.5:.1f}>"
        # so X.Y = (k+1)*0.5 => k = round(X.Y * 2) - 1
        val = float(d_tok[2:-1])  # strip "<d" and ">"
        k = round(val * 2) - 1
        if 0 <= k < NUM_DISTANCE_BINS:
            row = np.zeros(NUM_DISTANCE_BINS, dtype=np.float32)
            row[k] = 1.0
            probs[i - 1, j - 1, :] = row
            probs[j - 1, i - 1, :] = row

    for chunk_start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[chunk_start : chunk_start + batch_size]
        outputs = rt.llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for offset, gen in enumerate(outputs):
            lp_dict = gen.outputs[0].logprobs[0] if gen.outputs[0].logprobs else {}
            row = np.zeros(NUM_DISTANCE_BINS, dtype=np.float32)
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in distance_id_set:
                    row[bin_of[tid]] = math.exp(float(lp.logprob))
            total = float(row.sum())
            if total > 0:
                row /= total
            i, j = pairs[chunk_start + offset]
            probs[i - 1, j - 1, :] = row
            probs[j - 1, i - 1, :] = row
    return probs, len(pairs), len(prefix_ids)


def predict_one(
    *,
    rt: Runtime,
    stem: str,
    protenix_dir: Path,
    out_dir: Path,
    batch_size: int = 128,
    algorithm: str = "iterative_v1",
    n_rounds: int = 2,
    k_contacts_per_L_per_round: tuple[float, ...] = (1.0, 0.5),
    k_distances_per_L_per_round: tuple[float, ...] = (0.0, 1.0),
    min_contact_prob: float = 0.3,
    min_modal_prob: float = 0.5,
    sharpen_T_for_modes: float = 0.1,
    order: str = "long_med_short",
    initial_prior_path: Path | None = None,
) -> float:
    """Run R rounds of iterative seeding.

    Round 1 reads from ``initial_prior_path`` (default: ``distogram_baseline_naive.npz``).
    Each subsequent round reads from the previous round's output (in
    memory; we don't write intermediate distograms).
    """
    gt_cif = protenix_dir / "gt" / f"{stem}.cif"
    seq = read_canonical_sequence(gt_cif)
    pair_mask = build_gt_shell_mask(gt_cif, seq.n_residues)
    L = seq.n_residues

    if initial_prior_path is None:
        initial_prior_path = out_dir / stem / "distogram_baseline_naive.npz"
    current_prior = np.load(initial_prior_path)["probs"]

    t_start = time.time()
    rounds_meta: list[dict] = []
    already_committed: set[tuple[int, int]] = set()

    for r in range(n_rounds):
        k_contacts = int(round(k_contacts_per_L_per_round[r] * L))
        k_distances = int(round(k_distances_per_L_per_round[r] * L))
        contact_statements, distance_statements = extract_round_statements(
            current_prior,
            k_contacts=k_contacts,
            k_distances=k_distances,
            min_contact_prob=min_contact_prob,
            min_modal_prob=min_modal_prob,
            sharpen_T_for_modes=sharpen_T_for_modes,
            already_committed=already_committed,
            residue_names=seq.residue_names,
            order=order,
        )
        current_prior, n_pairs_queried, prefix_token_count = predict_distogram_with_statements(
            rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
            contact_statements=contact_statements,
            distance_statements=distance_statements,
            batch_size=batch_size,
        )
        for (i, j, *_r) in distance_statements:
            already_committed.add((i, j))
        rounds_meta.append({
            "round": r,
            "k_contacts": k_contacts,
            "k_distances": k_distances,
            "n_contact_statements": len(contact_statements),
            "n_distance_statements": len(distance_statements),
            "n_pairs_queried": n_pairs_queried,
            "prefix_token_count": prefix_token_count,
        })
    elapsed = time.time() - t_start

    n_pairs_total = L * (L - 1) // 2
    out_path = out_dir / stem / "distogram.npz"
    np.savez_compressed(out_path, probs=current_prior)
    (out_path.parent / "provenance.json").write_text(json.dumps({
        "stem": stem,
        "n_residues": L,
        "n_pairs": n_pairs_total,
        "n_pairs_queried": rounds_meta[-1]["n_pairs_queried"],
        "n_rounds": n_rounds,
        "rounds": rounds_meta,
        "min_contact_prob": min_contact_prob,
        "min_modal_prob": min_modal_prob,
        "sharpen_T_for_modes": sharpen_T_for_modes,
        "k_contacts_per_L_per_round": list(k_contacts_per_L_per_round),
        "k_distances_per_L_per_round": list(k_distances_per_L_per_round),
        "initial_prior_path": str(initial_prior_path),
        "algorithm": algorithm,
        "model_nickname": rt.model_nickname,
        "model_path": rt.model_path,
        "atom_convention": "CB-CB (CA for GLY/UNK)",
        "bin_scheme": {
            "min_A": 0.0,
            "max_A": DISTANCE_MAX_A,
            "n_bins": NUM_DISTANCE_BINS,
            "midpoints_A": BIN_MIDPOINTS.tolist(),
        },
        "elapsed_seconds": round(elapsed, 3),
        "model_load_seconds": round(rt.model_load_seconds, 3),
        "batch_size": batch_size,
        "hardware": rt.hardware,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    return elapsed
