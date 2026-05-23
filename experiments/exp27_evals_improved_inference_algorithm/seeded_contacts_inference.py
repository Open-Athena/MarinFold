# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Idea 1: self-bootstrapped contact seeding.

First pass: read distances naively (or, for speed, reuse the existing
``gt_filtered_naive`` distogram on disk). Convert to per-pair contact
probabilities (mass on bins ≤ 8 Å) and pick top-K most-confident
contacts subject to a minimum probability threshold. Second pass:
re-read distances on the LDDT-shell pairs with those contacts injected
into the ``<begin_statements>`` block, so the model conditions distance
predictions on its own most-confident contact commitments — the way
its training-time documents were structured.

**Seeds MUST come from a full naive distogram, not a gt_filtered
one.** Drawing seeds from a gt_filtered prior would inject GT
knowledge (we'd never seed a pair the GT says is out-of-shell),
giving us seeds with artificially high precision. The readout itself
may still be gt_filtered (reading at one pair is independent of
whether we queried other pairs — pure compute saving), but the prior
distogram must be the unfiltered baseline_naive output.

Use ``--prior-distogram-name distogram_baseline_naive.npz`` to point
at the snapshot of the full naive run.

Configurable knobs:
  --k-per-L         seeds count = round(k_per_L * L); default 1.0
  --min-contact-prob minimum contact prob to be eligible; default 0.5
  --order           one of {long_med_short, by_prob, random}
  --prior-algorithm name of the algorithm whose distograms to bootstrap
                    from; default ``gt_filtered_naive``.
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


# Match score_marinfold._CASP_SEPARATIONS — same separation buckets the
# scorer uses, same convention as vocab.CONTACT_TYPES comments.
_RANGE_BUCKETS = (
    # name, sep_lo, sep_hi (inclusive; None = open-ended), contact-token
    ("long",   24, None, "<long-range-contact>"),
    ("medium", 12, 23,   "<medium-range-contact>"),
    ("short",  6,  11,   "<short-range-contact>"),
)
_CONTACT_BIN_MAX_A = 8.0
_CONTACT_BIN_COUNT = int(_CONTACT_BIN_MAX_A / 0.5)  # 16 bins covering 0..8 Å


def extract_seed_contacts(
    probs: np.ndarray,
    *,
    k_total: int,
    min_contact_prob: float,
    order: str,
    rng: np.random.Generator,
) -> list[tuple[int, int, str]]:
    """Pick up to ``k_total`` seed contacts from a prior distogram.

    Returns ``[(i, j, "<{range}-range-contact>")]`` with 1-indexed
    positions. Filters by contact-probability threshold first, then
    sorts according to ``order`` and truncates.
    """
    n = probs.shape[0]
    contact_probs = probs[:, :, :_CONTACT_BIN_COUNT].sum(axis=-1)
    candidates: list[tuple[int, int, str, float]] = []
    for name, sep_lo, sep_hi, token in _RANGE_BUCKETS:
        for i in range(n):
            j_start = i + sep_lo
            j_end = n if sep_hi is None else min(n, i + sep_hi + 1)
            for j in range(j_start, j_end):
                p = float(contact_probs[i, j])
                if p < min_contact_prob:
                    continue
                candidates.append((i + 1, j + 1, token, p))

    if not candidates:
        return []

    if order == "long_med_short":
        # Stable sort by (range_index, -prob). long first matches
        # vocab.CONTACT_TYPES ordering.
        range_index = {tok: idx for idx, (_, _, _, tok) in enumerate(_RANGE_BUCKETS)}
        candidates.sort(key=lambda c: (range_index[c[2]], -c[3]))
    elif order == "by_prob":
        candidates.sort(key=lambda c: -c[3])
    elif order == "random":
        rng.shuffle(candidates)
    else:
        raise ValueError(f"unknown order: {order}")

    chosen = candidates[:k_total]
    return [(i, j, tok) for (i, j, tok, _p) in chosen]


def predict_distogram_seeded(
    *,
    rt: Runtime,
    residue_names: list[str],
    pair_mask: np.ndarray,
    seeds: list[tuple[int, int, str]],
    batch_size: int = 128,
    top_k_logprobs: int = 128,
) -> tuple[np.ndarray, int, int]:
    """Read distances at pairs in ``pair_mask`` with contact ``seeds`` in the prefix.

    Returns ``(probs, n_pairs_queried, prefix_token_count)``.
    """
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    if pair_mask.shape != (n, n):
        raise ValueError(f"pair_mask shape {pair_mask.shape} != ({n}, {n})")

    base_tokens = _build_base_prompt(residue_names)
    seed_tokens: list[str] = []
    for (i, j, tok) in seeds:
        seed_tokens.extend([tok, f"<p{i}>", f"<p{j}>"])
    prefix_tokens = base_tokens + seed_tokens
    prefix_ids = _encode_tokens(rt.tokenizer, prefix_tokens)

    distance_id_set = set(rt.distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(rt.distance_token_ids)}

    pairs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if pair_mask[i - 1, j - 1]:
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
    algorithm: str = "seeded_contacts_v1",
    k_per_L: float = 1.0,
    min_contact_prob: float = 0.5,
    order: str = "long_med_short",
    prior_distogram_path: Path | None = None,
    seed_rng: int = 27,
) -> float:
    """Predict distances with self-bootstrapped contact seeds.

    The prior distogram path defaults to ``out_dir/<stem>/distogram.npz``
    (whatever was last written there). If that's a gt_filtered distogram,
    seeds will be drawn from in-shell pairs only — that's intentional
    for this experiment dir (everything here is gt_filtered-keyed).
    """
    gt_cif = protenix_dir / "gt" / f"{stem}.cif"
    seq = read_canonical_sequence(gt_cif)
    pair_mask = build_gt_shell_mask(gt_cif, seq.n_residues)

    if prior_distogram_path is None:
        prior_distogram_path = out_dir / stem / "distogram.npz"
    prior = np.load(prior_distogram_path)["probs"]
    if prior.shape[0] != seq.n_residues:
        raise ValueError(
            f"prior distogram for {stem} has N={prior.shape[0]} but "
            f"sequence has N={seq.n_residues}"
        )

    k_total = max(1, round(k_per_L * seq.n_residues))
    rng = np.random.default_rng(seed_rng)
    seeds = extract_seed_contacts(
        prior,
        k_total=k_total,
        min_contact_prob=min_contact_prob,
        order=order,
        rng=rng,
    )

    t_start = time.time()
    probs, n_pairs_queried, prefix_token_count = predict_distogram_seeded(
        rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
        seeds=seeds, batch_size=batch_size,
    )
    elapsed = time.time() - t_start

    n_pairs_total = seq.n_residues * (seq.n_residues - 1) // 2
    out_path = out_dir / stem / "distogram.npz"
    np.savez_compressed(out_path, probs=probs)
    (out_path.parent / "provenance.json").write_text(json.dumps({
        "stem": stem,
        "n_residues": seq.n_residues,
        "n_pairs": n_pairs_total,
        "n_pairs_queried": n_pairs_queried,
        "pair_filter_fraction": round(n_pairs_queried / n_pairs_total, 4),
        "n_seed_contacts": len(seeds),
        "k_per_L": k_per_L,
        "min_contact_prob": min_contact_prob,
        "seed_order": order,
        "seed_rng": seed_rng,
        "prefix_token_count": prefix_token_count,
        "prior_distogram_path": str(prior_distogram_path),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    parser.add_argument(
        "--protenix-dir", type=Path,
        default=_THIS / "protenix_data" / "data" / "protenix-foldbench-monomers",
    )
    parser.add_argument("--out", type=Path, default=_THIS / "outputs")
    parser.add_argument("--model", default="1B")
    parser.add_argument(
        "--models-yaml", type=Path,
        default=_THIS.parent.parent / "MODELS.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--k-per-L", type=float, default=1.0)
    parser.add_argument("--min-contact-prob", type=float, default=0.5)
    parser.add_argument(
        "--order",
        default="long_med_short",
        choices=["long_med_short", "by_prob", "random"],
    )
    args = parser.parse_args()
    rt = load_runtime(
        model_nickname=args.model, models_yaml=args.models_yaml,
        dtype=args.dtype,
    )
    elapsed = predict_one(
        rt=rt, stem=args.stem,
        protenix_dir=args.protenix_dir, out_dir=args.out,
        batch_size=args.batch_size,
        k_per_L=args.k_per_L,
        min_contact_prob=args.min_contact_prob,
        order=args.order,
    )
    print(f"{args.stem}: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
