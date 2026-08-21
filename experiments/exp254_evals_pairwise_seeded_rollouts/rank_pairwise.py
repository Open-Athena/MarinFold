# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 -- the pairwise ``P(contact)`` readout, and the per-protein seed list.

For each eval protein this computes the contacts-v1 *pairwise* readout on a
canonical document realization:

    lp1[i]    = log P(<p_i> | prefix, <contact>)
    lp2[i, j] = log P(<p_j> | prefix, <contact>, <p_i>)
    P(contact)[i, j] = exp(lp1[i] + lp2[i, j]) + exp(lp1[j] + lp2[j, i])

which is exactly ``marinfold.document_structures.contacts_v1.inference``'s
``_pcontact_matrix``, re-implemented against a raw vLLM handle so that the same
engine can serve phase 2's rollouts without a second model load.

Two outputs:

* ``seeds.parquet`` -- the top ``--n-seeds`` pairs per protein (``sep >= 6``,
  upper triangle), ranked by ``P(contact)``. Phase 2's ``seeded`` arm prompts
  rollout *r* with rank *r*.
* ``pairwise/<dataset>__<stem>.npz`` -- the dense symmetric ``[L, L]`` matrix, so
  the pairwise readout itself can be scored as a predictor alongside the rollout
  arms. This is the readout exp82 superseded; it is here for context, never as
  the frontier number.

**Full-vocab logprobs, not top-k.** contacts-v1 has a 2,845-token vocabulary, so
asking vLLM for a logprob per vocabulary entry is cheap and makes the readout
exact. The usual top-k truncation (which silently zeroes any position token
outside the kept set) would put an arbitrary floor under the tail of the ranking
and is asserted against here.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import MIN_SEP, EXPECTED_UNITS, load_targets, realization

SEED_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rank", pa.int16()), ("i", pa.int16()), ("j", pa.int16()),
    ("p_contact", pa.float64()), ("range", pa.string()),
])

#: CASP sequence-separation bins, ``(lo, hi)`` inclusive, ``hi=None`` unbounded.
#: These must stay identical to the ``short`` / ``medium`` / ``long`` entries of
#: exp89's ``RANGES`` -- a seed labelled ``long`` here has to be the same thing
#: the long-range metric scores, or the stratified arms answer a question about
#: a different partition than the one being reported. ``test_exp254.py`` pins it.
SEED_RANGES: dict[str, tuple[int, int | None]] = {
    "short": (6, 11), "medium": (12, 23), "long": (24, None),
}

#: Probability floor before taking a log, matching contacts_v1.inference.
PROB_FLOOR = 1e-30


def token_id(tokenizer, token: str) -> int:
    """The single vocabulary id for ``token``, or a hard failure."""
    tid = tokenizer.convert_tokens_to_ids(token)
    assert tid is not None and tid >= 0, (
        f"tokenizer has no dedicated id for {token!r}: it is missing the "
        f"contacts-v1 vocabulary"
    )
    return int(tid)


def pcontact_matrix(llm, sampling, tokenizer, prefix: str, seq_positions: list[int]):
    """``P(contact)`` over every residue pair for one realization.

    One vLLM call carries all ``L + 1`` prompts: the ``<contact>`` continuation
    that gives ``lp1`` and the ``L`` ``<contact> <p_i>`` tails that give ``lp2``.
    vLLM's prefix cache makes the shared sequence-section prefix free after the
    first.
    """
    from vllm import TokensPrompt

    from marinfold.document_structures.contacts_v1.vocab import (
        CONTACT_TOKEN,
        position_token,
    )

    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    contact_id = token_id(tokenizer, CONTACT_TOKEN)
    pos_ids = [token_id(tokenizer, position_token(p)) for p in seq_positions]
    col_of = {tid: c for c, tid in enumerate(pos_ids)}

    prompts = [TokensPrompt(prompt_token_ids=prefix_ids + [contact_id])]
    prompts += [
        TokensPrompt(prompt_token_ids=prefix_ids + [contact_id, pid])
        for pid in pos_ids
    ]
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    n = len(pos_ids)
    probs = np.zeros((len(prompts), n), dtype=np.float64)
    for row, result in enumerate(outputs):
        lp_dict = result.outputs[0].logprobs[0]
        n_hit = 0
        for tok, lp in lp_dict.items():
            col = col_of.get(int(tok))
            if col is not None:
                probs[row, col] = float(np.exp(float(lp.logprob)))
                n_hit += 1
        assert n_hit == n, (
            f"logprobs covered {n_hit}/{n} position tokens -- the readout is "
            f"top-k truncated; raise max_logprobs to the full vocabulary"
        )

    lp1 = np.log(np.clip(probs[0], PROB_FLOOR, None))
    lp2 = np.log(np.clip(probs[1:], PROB_FLOOR, None))
    fwd = lp1[:, None] + lp2                       # log P(i) * P(j | i)
    return np.exp(fwd) + np.exp(fwd.T)             # unordered, symmetric


def range_of(separation: np.ndarray) -> np.ndarray:
    """CASP bin label for each sequence separation."""
    return np.where(separation >= 24, "long",
                    np.where(separation >= 12, "medium", "short"))


def top_pairs(matrix: np.ndarray, n_seeds: int, min_sep: int,
              *, band: tuple[int, int | None] | None = None):
    """The ``n_seeds`` highest-scoring ``(i, j)`` pairs with ``j - i >= min_sep``.

    ``band`` optionally restricts the candidates to one inclusive separation
    window, which is what the stratified and long-range strategies select
    within. Ties are broken by the pair order ``np.argsort`` produces, which is
    stable, so the seed list is a deterministic function of the matrix.
    """
    L = matrix.shape[0]
    ii, jj = np.triu_indices(L, k=min_sep)
    if band is not None:
        lo, hi = band
        separation = jj - ii
        inside = separation >= max(lo, min_sep)
        if hi is not None:
            inside = inside & (separation <= hi)
        ii, jj = ii[inside], jj[inside]
    scores = matrix[ii, jj]
    keep = min(n_seeds, scores.size)
    order = np.argsort(-scores, kind="mergesort")[:keep]
    return ii[order], jj[order], scores[order]


def stratum_quotas(n_seeds: int) -> dict[str, int]:
    """Split ``n_seeds`` across the three CASP bins as evenly as possible.

    100 does not divide by three. The remainder goes to the longest bins first
    (long, then medium), because long-range contacts are where exp254 found the
    only remaining best-of-N headroom -- so 100 seeds are 34 long / 33 medium /
    33 short.
    """
    base, remainder = divmod(n_seeds, len(SEED_RANGES))
    quotas = {name: base for name in SEED_RANGES}
    for name in ("long", "medium", "short")[:remainder]:
        quotas[name] += 1
    return quotas


def select_seeds(matrix: np.ndarray, n_seeds: int, min_sep: int, strategy: str):
    """Seed list for one protein under one selection strategy.

    ``top``
        the ``n_seeds`` best pairs overall. On this eval set that is already
        **56.8 % long-range**, because long-separation pairs dominate the
        candidate universe -- so "top-N" is not the short-range-heavy list it
        sounds like, and equal thirds is a long-range *reduction*, not a bias.
    ``stratified``
        the best ``n_seeds / 3`` within each CASP bin, round-robin interleaved
        so consecutive rollouts cover different bins. Every one of eval-val's 97
        proteins has enough candidates in every bin (the smallest, L=38, has
        177 / 246 / 105), so no bin is ever short and no top-up rule is needed.
    ``long``
        all ``n_seeds`` drawn from ``sep >= 24``. This is the strategy that
        actually biases toward long range.

    Returns ``(i, j, score, range_label)``.
    """
    if strategy == "top":
        ii, jj, scores = top_pairs(matrix, n_seeds, min_sep)
        return ii, jj, scores, range_of(jj - ii)
    if strategy == "long":
        ii, jj, scores = top_pairs(matrix, n_seeds, min_sep,
                                   band=SEED_RANGES["long"])
        assert len(ii) == n_seeds, (
            f"only {len(ii)} long-range candidates for {n_seeds} seeds"
        )
        return ii, jj, scores, range_of(jj - ii)
    if strategy != "stratified":
        raise ValueError(f"unknown seed strategy {strategy!r}")

    quotas = stratum_quotas(n_seeds)
    per_bin = {}
    for name, band in SEED_RANGES.items():
        ii, jj, scores = top_pairs(matrix, quotas[name], min_sep, band=band)
        assert len(ii) == quotas[name], (
            f"only {len(ii)} candidates in the {name} bin for "
            f"{quotas[name]} seeds"
        )
        per_bin[name] = (ii, jj, scores)

    # Round-robin, longest bin first: rollout r's bin is r % 3, so a partial run
    # is balanced across bins and the rank-versus-quality curve stays readable.
    order, cursors = [], {name: 0 for name in per_bin}
    while len(order) < n_seeds:
        for name in ("long", "medium", "short"):
            if cursors[name] < quotas[name] and len(order) < n_seeds:
                order.append((name, cursors[name]))
                cursors[name] += 1
    ii = np.array([per_bin[n][0][k] for n, k in order])
    jj = np.array([per_bin[n][1][k] for n, k in order])
    scores = np.array([per_bin[n][2][k] for n, k in order])
    labels = np.array([n for n, _ in order])
    return ii, jj, scores, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--strategy", choices=("top", "stratified", "long"),
                    default="top",
                    help="how the n seeds are drawn from the pairwise matrix; "
                         "the seed file is named seeds_<strategy>.parquet")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    targets = load_targets()
    assert len(targets) == EXPECTED_UNITS, (
        f"expected {EXPECTED_UNITS} eval-val units, got {len(targets)}"
    )
    todo = targets[: args.limit] if args.limit else targets

    matrix_dir = args.out / "pairwise"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = len(tokenizer)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=args.gpu_frac, enable_prefix_caching=True,
              generation_config="vllm", max_logprobs=vocab_size, seed=0)
    # temperature 1 / top-p 1 / top-k off: the returned distribution is the
    # model's own, undistorted by sampling knobs.
    sampling = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1,
                              logprobs=vocab_size, n=1)

    from marinfold.document_structures.contacts_v1 import residues_from_sequence

    rows = {c: [] for c in
            ("dataset", "stem", "L", "rank", "i", "j", "p_contact", "range")}
    t0 = time.time()
    for n, target in enumerate(todo, start=1):
        residues = residues_from_sequence(target.input_seq)
        prefix, seq_positions = realization(target.stem, residues, "pw")
        assert len(seq_positions) == target.L, (
            f"{target.stem}: realization length {len(seq_positions)} != L={target.L}"
        )
        matrix = pcontact_matrix(llm, sampling, tokenizer, prefix, seq_positions)
        np.savez_compressed(matrix_dir / f"{target.key}.npz",
                            score=matrix.astype(np.float32))
        ii, jj, scores, labels = select_seeds(matrix, args.n_seeds, MIN_SEP,
                                              args.strategy)
        rows["dataset"] += [target.dataset] * len(ii)
        rows["stem"] += [target.stem] * len(ii)
        rows["L"] += [target.L] * len(ii)
        rows["rank"] += list(range(len(ii)))
        rows["i"] += ii.astype(np.int16).tolist()
        rows["j"] += jj.astype(np.int16).tolist()
        rows["p_contact"] += scores.tolist()
        rows["range"] += labels.tolist()
        print(f"[pairwise] [{n}/{len(todo)}] {target.stem} L={target.L} "
              f"seeds={len(ii)} top_p={scores[0]:.4g} "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    dest = args.out / f"seeds_{args.strategy}.parquet"
    pq.write_table(pa.table(rows, schema=SEED_SCHEMA), dest, compression="zstd")
    composition = pd.Series(rows["range"]).value_counts(normalize=True) * 100
    print(f"[pairwise] wrote {len(rows['stem'])} seed rows for {len(todo)} "
          f"proteins (strategy={args.strategy}) -> {dest}")
    print("[pairwise] seed composition by separation range (%):")
    print(composition.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
