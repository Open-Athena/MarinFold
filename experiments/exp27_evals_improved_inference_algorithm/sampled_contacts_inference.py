# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Idea 6: model-sampled contact prefix.

Honest version of idea 1 — instead of picking seed contacts from a
prior naive distogram (which requires a full naive pass), let the
model autoregressively generate a contact-statement block from the
base prompt ``<begin_sequence><AAs><begin_statements>``. Parse the
emitted statements (stop at ``<distance>``, ``<end>``, max-token cap,
or first malformed sequence), use them as the prefix for a distance
readout on LDDT-shell pairs.

This matches the training distribution: docs were generated as
``[contacts]* [distances]*``, so sampling from ``<begin_statements>``
should produce contacts naturally.

For multi-rollout averaging, run M independent samples (different
``seed`` values), each gets its own readout, average final distograms.
"""

import argparse
import json
import math
import re
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


_CONTACT_TOKEN_STRS = (
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
)
_POSITION_RE = re.compile(r"<p(\d+)>")
_CONTACT_RE = re.compile(
    r"<(long|medium|short)-range-contact>\s*<p(\d+)>\s*<p(\d+)>"
)


def _resolve_contact_range_ids(tokenizer) -> list[int]:
    ids = []
    for tok in _CONTACT_TOKEN_STRS:
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"contact token {tok!r} didn't tokenize to 1: {enc!r}")
        ids.append(enc[0])
    return ids


def _resolve_position_token_ids(tokenizer, n_residues: int) -> list[int]:
    """Return IDs for ``<p1>`` .. ``<p_{n_residues}>``.

    Positions are 1-indexed in the v1 grammar (matches the canonical
    1..N residue numbering).
    """
    ids = []
    for i in range(1, n_residues + 1):
        enc = tokenizer.encode(f"<p{i}>", add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"position <p{i}> didn't tokenize to 1: {enc!r}")
        ids.append(enc[0])
    return ids


class ContactsOnlyLogitsProcessor:
    """Force vLLM sampling to emit only contact statements.

    Training documents stochastically interleave contact and distance
    statements, so the model emits ``<distance>`` after a few contacts
    on its own. To get many sampled contacts per rollout, we mask the
    logits to a 3-state cycle:

      state 0 (modulo 3): allow only the 3 ``<*-range-contact>`` tokens
      state 1            : allow only position tokens ``<p1>`` .. ``<pN>``
      state 2            : same as state 1

    ``range_strategy`` controls how the contact-range token is chosen
    at state 0:

      "model"       : keep the model's logits over the 3 range tokens.
                      In practice the model strongly prefers
                      ``<medium-range-contact>`` (~99% at T=0.7).
      "uniform"     : overwrite the 3 logits to 0 so the softmax is
                      uniform → ~1/3 of each range, sampled randomly.
      "round_robin" : deterministic L → M → S cycle. Exactly 1/3 of
                      each range over a long-enough rollout.

    Validity of (i, j) pairs is checked in post-processing — the LP
    only constrains token *type* / *range-token identity*, not pair
    semantics.
    """

    def __init__(
        self,
        *,
        contact_range_ids: list[int],
        position_ids: list[int],
        vocab_size: int,
        range_strategy: str = "model",
    ):
        # ordered as [long, medium, short] — matches _CONTACT_TOKEN_STRS
        self._contact_ids_ordered = list(contact_range_ids)
        self._range_strategy = range_strategy
        self._vocab_size = vocab_size
        # Pre-build masks lazily on the GPU device the first time __call__ is hit.
        # Pre-building eagerly on CPU + moving every call adds ~ms per token,
        # which dominates wall time when the LP runs 9000+ times per protein.
        self._contact_range_ids_t = contact_range_ids
        self._position_ids_t = position_ids
        self._device_masks = None  # populated on first __call__

    def _materialise(self, device, vocab_size_runtime):
        import torch
        all_contact = torch.full((vocab_size_runtime,), float("-inf"), device=device)
        for tid in self._contact_range_ids_t:
            if tid < vocab_size_runtime:
                all_contact[tid] = 0.0
        position = torch.full((vocab_size_runtime,), float("-inf"), device=device)
        for tid in self._position_ids_t:
            if tid < vocab_size_runtime:
                position[tid] = 0.0
        single = []
        for tid in self._contact_range_ids_t:
            m = torch.full((vocab_size_runtime,), float("-inf"), device=device)
            if tid < vocab_size_runtime:
                m[tid] = 0.0
            single.append(m)
        uniform = torch.full((vocab_size_runtime,), float("-inf"), device=device)
        for tid in self._contact_range_ids_t:
            if tid < vocab_size_runtime:
                uniform[tid] = 0.0
        # "uniform" is the OVERWRITE-style mask: every allowed token gets logit 0,
        # blocked tokens are -inf; we return this directly (not added to logits)
        # so the model's prior is ignored.
        self._device_masks = {
            "all_contact": all_contact,
            "position": position,
            "single": single,
            "uniform": uniform,
        }

    def __call__(self, generated_token_ids, logits):
        if self._device_masks is None:
            self._materialise(logits.device, logits.shape[-1])
        m = self._device_masks
        state = len(generated_token_ids) % 3
        if state == 0:
            if self._range_strategy == "model":
                return logits + m["all_contact"]
            if self._range_strategy == "uniform":
                return m["uniform"]
            if self._range_strategy == "round_robin":
                round_idx = len(generated_token_ids) // 3
                return m["single"][round_idx % 3]
            raise ValueError(f"unknown range_strategy: {self._range_strategy}")
        return logits + m["position"]


def _range_token_for_separation(sep: int) -> str | None:
    """Return the contact-range token matching this sequence separation.

    Matches score_marinfold._CASP_SEPARATIONS:
      short  = 6..11   medium = 12..23   long = 24+
    Returns None for sep < 6 (invalid for contacts).
    """
    if sep < 6:
        return None
    if sep <= 11:
        return "<short-range-contact>"
    if sep <= 23:
        return "<medium-range-contact>"
    return "<long-range-contact>"


def sample_contact_prefix(
    rt: Runtime,
    residue_names: list[str],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    range_strategy: str = "model",
) -> tuple[list[tuple[int, int, str]], int]:
    """Sample one contact-only rollout from ``<begin_statements>``.

    Uses a :class:`ContactsOnlyLogitsProcessor` to mask all non-contact
    tokens during generation, so the entire ``max_tokens`` budget gets
    spent emitting contact statements (rather than the model
    transitioning to ``<distance>`` after a handful).

    Post-processing:
      * drop pairs with i == j, position out of [1, N], or |i-j| < 6
        (CASP minimum for a "real" contact)
      * dedupe by (i, j) — keep first occurrence so the model's
        token-order ranking is preserved
      * **rewrite the range token to match the actual |i-j|** — the
        model can sample (e.g.) ``<short-range-contact>`` followed by
        positions 100 apart; we always reseed with the bucket the
        scorer uses

    Returns ``(seeds, n_decoded_tokens)``.
    """
    from vllm import SamplingParams, TokensPrompt

    base_tokens = _build_base_prompt(residue_names)
    base_ids = _encode_tokens(rt.tokenizer, base_tokens)
    n = len(residue_names)

    vocab_size = rt.tokenizer.vocab_size
    lp = ContactsOnlyLogitsProcessor(
        contact_range_ids=_resolve_contact_range_ids(rt.tokenizer),
        position_ids=_resolve_position_token_ids(rt.tokenizer, n),
        vocab_size=vocab_size,
        range_strategy=range_strategy,
    )
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=-1,
        max_tokens=max_tokens,
        n=1,
        seed=seed,
        logits_processors=[lp],
    )
    outputs = rt.llm.generate(
        [TokensPrompt(prompt_token_ids=base_ids)], sampling, use_tqdm=False,
    )
    gen_ids = list(outputs[0].outputs[0].token_ids)
    gen_text = rt.tokenizer.decode(gen_ids, skip_special_tokens=False)

    seeds: list[tuple[int, int, str]] = []
    for m in _CONTACT_RE.finditer(gen_text):
        i = int(m.group(2))
        j = int(m.group(3))
        if i < 1 or j < 1 or i > n or j > n or i == j:
            continue
        if i > j:
            i, j = j, i
        tok = _range_token_for_separation(j - i)
        if tok is None:
            continue
        seeds.append((i, j, tok))
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int, str]] = []
    for (i, j, tok) in seeds:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        deduped.append((i, j, tok))
    return deduped, len(gen_ids)


def predict_distogram_with_prefix(
    *,
    rt: Runtime,
    residue_names: list[str],
    pair_mask: np.ndarray,
    seeds: list[tuple[int, int, str]],
    batch_size: int = 128,
    top_k_logprobs: int = 128,
) -> tuple[np.ndarray, int, int]:
    """Same as seeded_contacts_inference.predict_distogram_seeded.

    Inlined here so this module is self-contained (the algorithm-side
    contract is the only thing that changes between ideas).
    """
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    if pair_mask.shape != (n, n):
        raise ValueError(f"pair_mask shape {pair_mask.shape} != ({n}, {n})")

    base_tokens = _build_base_prompt(residue_names)
    seed_tokens: list[str] = []
    for (i, j, tok) in seeds:
        seed_tokens.extend([tok, f"<p{i}>", f"<p{j}>"])
    prefix_ids = _encode_tokens(rt.tokenizer, base_tokens + seed_tokens)

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
    algorithm: str = "sampled_contacts_v1",
    n_rollouts: int = 1,
    max_sample_tokens: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.9,
    base_seed: int = 27,
    aggregation: str = "average",
    range_strategy: str = "model",
) -> float:
    """Predict distances with model-sampled contact prefixes.

    ``aggregation`` controls how multiple rollouts are combined:
      - ``average``: each rollout gets its own readout; per-pair probs
        are averaged. (Earlier finding: this *hurts* — blurs the
        distogram.)
      - ``union``: take the UNION of sampled seed contacts across
        rollouts (deduplicated by (i, j)), then do a single readout
        with the union as the prefix. Aims to enrich the seed set with
        diverse low-frequency contacts that any single rollout would
        miss.
    """
    gt_cif = protenix_dir / "gt" / f"{stem}.cif"
    seq = read_canonical_sequence(gt_cif)
    pair_mask = build_gt_shell_mask(gt_cif, seq.n_residues)

    t_start = time.time()
    rollouts_meta: list[dict] = []
    n_pairs_queried = 0
    prefix_token_count = 0

    if aggregation == "average":
        accumulator = np.zeros(
            (seq.n_residues, seq.n_residues, NUM_DISTANCE_BINS), dtype=np.float32,
        )
        for r in range(n_rollouts):
            seeds, n_sample_tokens = sample_contact_prefix(
                rt, seq.residue_names,
                max_tokens=max_sample_tokens,
                temperature=temperature, top_p=top_p,
                seed=base_seed + r,
                range_strategy=range_strategy,
            )
            probs_r, n_pairs_queried, prefix_token_count = predict_distogram_with_prefix(
                rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
                seeds=seeds, batch_size=batch_size,
            )
            accumulator += probs_r
            rollouts_meta.append({
                "rollout": r, "seed": base_seed + r,
                "n_sample_tokens": n_sample_tokens,
                "n_seeds": len(seeds),
                "prefix_token_count": prefix_token_count,
            })
        probs = accumulator / max(1, n_rollouts)
        row_sums = probs.sum(axis=-1, keepdims=True)
        np.divide(probs, row_sums, out=probs, where=row_sums > 0)
    elif aggregation == "union":
        all_seeds: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for r in range(n_rollouts):
            seeds, n_sample_tokens = sample_contact_prefix(
                rt, seq.residue_names,
                max_tokens=max_sample_tokens,
                temperature=temperature, top_p=top_p,
                seed=base_seed + r,
                range_strategy=range_strategy,
            )
            rollouts_meta.append({
                "rollout": r, "seed": base_seed + r,
                "n_sample_tokens": n_sample_tokens,
                "n_seeds": len(seeds),
            })
            for (i, j, tok) in seeds:
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                all_seeds.append((i, j, tok))
        # Sort the union long → medium → short for prefix
        rng_idx = {
            "<long-range-contact>": 0,
            "<medium-range-contact>": 1,
            "<short-range-contact>": 2,
        }
        all_seeds.sort(key=lambda s: (rng_idx[s[2]], s[0], s[1]))
        probs, n_pairs_queried, prefix_token_count = predict_distogram_with_prefix(
            rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
            seeds=all_seeds, batch_size=batch_size,
        )
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")
    elapsed = time.time() - t_start

    n_pairs_total = seq.n_residues * (seq.n_residues - 1) // 2
    out_path = out_dir / stem / "distogram.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, probs=probs)
    (out_path.parent / "provenance.json").write_text(json.dumps({
        "stem": stem,
        "n_residues": seq.n_residues,
        "n_pairs": n_pairs_total,
        "n_pairs_queried": n_pairs_queried,
        "pair_filter_fraction": round(n_pairs_queried / n_pairs_total, 4),
        "n_rollouts": n_rollouts,
        "aggregation": aggregation,
        "max_sample_tokens": max_sample_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": base_seed,
        "range_strategy": range_strategy,
        "rollouts": rollouts_meta,
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
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--max-sample-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--base-seed", type=int, default=27)
    args = parser.parse_args()
    rt = load_runtime(
        model_nickname=args.model, models_yaml=args.models_yaml,
        dtype=args.dtype,
    )
    elapsed = predict_one(
        rt=rt, stem=args.stem,
        protenix_dir=args.protenix_dir, out_dir=args.out,
        batch_size=args.batch_size,
        n_rollouts=args.n_rollouts,
        max_sample_tokens=args.max_sample_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        base_seed=args.base_seed,
    )
    print(f"{args.stem}: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
