# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 Inference (predict + evaluate).

Loads from this file via ``get_inference()``. Powers two
``marinfold-document-structure`` subcommands:

- ``infer    <this_dir> --model M --input X     --out preds.parquet``
- ``evaluate <this_dir> --model M --input X     --out metrics.json``

Both walk every residue pair (i, j) with i < j on every input
structure, query the model at the prompt

    <contacts-and-distances-v1> <begin_sequence> <AAs>
    <begin_statements>
    [N seeded <long-range-contact> <p_i> <p_j> triples (evaluate only)]
    <distance> <p_i> <p_j> <atom_i> <atom_j>

and renormalize the next-token distribution over the 64 ``<d_X.X>``
bins to an expected distance.

- ``infer`` yields one record per input structure: the entry id, the
  pairs queried, and the expected distance per pair. No ground truth
  is consulted; predictions for any structure (including ones where
  you only have the sequence) are valid output.

- ``evaluate`` additionally computes |expected − GT_distance| per
  pair, where the GT distance is taken from the same input file
  (the inputs ARE the ground truth in this mode), and reports the
  macro-mean MAE across structures. ``--seed-n-values 0,5,20,50``
  sweeps the seeded-contact count in a single run (matches Phase 7c
  from the LlamaFold-experiments notebook).

Inspired by marin's experiments/protein/eval_protein_distogram.py.
"""

import argparse
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marinfold_document_structures import EvalResult

from _parse import (
    ParsedStructure,
    Residue,
    atom_position,
    cb_or_ca_position,
    euclidean,
    iter_parsed_structures,
)
from _vocab import CONTEXT_LENGTH, NAME, all_domain_tokens


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


# Distance-bin midpoints (Å) — bin k (1..64) covers ((k-1)*0.5,
# k*0.5] Å, so its midpoint is k*0.5 - 0.25. Matches
# marin's eval_protein_distogram.py.
_DISTANCE_BIN_WIDTH_A = 0.5
_NUM_DISTANCE_BINS = 64
_DISTANCE_MAX_A = _NUM_DISTANCE_BINS * _DISTANCE_BIN_WIDTH_A  # 32.0
_DISTANCE_BIN_MIDPOINTS = tuple(
    (k + 1) * _DISTANCE_BIN_WIDTH_A - _DISTANCE_BIN_WIDTH_A / 2
    for k in range(_NUM_DISTANCE_BINS)
)

# Contact-eligibility cutoffs for the "seeded GT contacts" hint mode.
# Match the v1 format definition.
_CONTACT_CUTOFF_A = 8.0
_LONG_RANGE_SEP = 24


def _parse_seed_n_values(s: str) -> tuple[int, ...]:
    """Parse `--seed-n-values 0,5,20,50` into a tuple of ints."""
    out: list[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        n = int(token)
        if n < 0:
            raise argparse.ArgumentTypeError(f"seed-n value must be >= 0; got {n}")
        out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("--seed-n-values must include at least one value")
    return tuple(out)


# --------------------------------------------------------------------------
# Prompt + GT helpers
# --------------------------------------------------------------------------


def _gt_query_distance_matrix(structure: ParsedStructure, atom_name: str):
    """N×N distance matrix using ``atom_name`` on both endpoints.

    Returns NaN for residues missing the named atom.
    """
    import numpy as np

    n = len(structure.residues)
    positions = [atom_position(r, atom_name) for r in structure.residues]
    out = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        pi = positions[i]
        if pi is None:
            continue
        for j in range(n):
            pj = positions[j]
            if pj is None:
                continue
            if i == j:
                out[i, j] = 0.0
                continue
            out[i, j] = float(np.linalg.norm(np.asarray(pi) - np.asarray(pj)))
    return out


def _gt_long_range_contacts(structure: ParsedStructure) -> list[tuple[int, int]]:
    """1-indexed (i, j) with CB-CB ≤ 8 Å, sep ≥ 24, sorted by (i, j)."""
    cb: dict[int, tuple[float, float, float]] = {}
    for r in structure.residues:
        p = cb_or_ca_position(r)
        if p is not None:
            cb[r.index] = p
    out: list[tuple[int, int]] = []
    indices = sorted(cb)
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            i, j = indices[ii], indices[jj]
            if j - i < _LONG_RANGE_SEP:
                continue
            if euclidean(cb[i], cb[j]) <= _CONTACT_CUTOFF_A:
                out.append((i, j))
    out.sort()
    return out


def _build_base_prompt_tokens(
    structure: ParsedStructure,
    seeded_contacts: list[tuple[int, int]],
) -> list[str]:
    """Base prompt = `<task> <begin_sequence> <AAs> <begin_statements> [seeded contacts]`."""
    toks: list[str] = [f"<{NAME}>", "<begin_sequence>"]
    toks.extend(f"<{r.name}>" for r in structure.residues)
    toks.append("<begin_statements>")
    for i, j in seeded_contacts:
        toks.append("<long-range-contact>")
        toks.append(f"<p{i}>")
        toks.append(f"<p{j}>")
    return toks


def _pair_tail_tokens(i: int, j: int, atom_i: str, atom_j: str) -> list[str]:
    """5-token tail that elicits a `<d_X.X>` next-token distribution."""
    return ["<distance>", f"<p{i}>", f"<p{j}>", f"<{atom_i}>", f"<{atom_j}>"]


def _resolve_distance_token_ids(tokenizer) -> list[int]:
    """The 64 distance-bin token IDs, in bin order (k=0 → `<d0.5>`)."""
    ids: list[int] = []
    for k in range(_NUM_DISTANCE_BINS):
        tok = f"<d{(k + 1) * _DISTANCE_BIN_WIDTH_A:.1f}>"
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"Unexpected encoding for {tok}: {enc!r}")
        ids.append(int(enc[0]))
    # WordLevel tokenizers return the same unk_token_id for every
    # unknown string, so the per-token len==1 check above silently
    # passes even when every <d_X.X> is UNK. Catch that explicitly:
    # the failure mode otherwise is a constant ~31.75 Å expected
    # distance and ~14 Å MAE that looks like a badly-trained model.
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and unk_id in ids:
        raise ValueError(
            f"Tokenizer mapped one or more <d_X.X> bin tokens to the "
            f"UNK id ({unk_id}). The tokenizer is missing the v1 "
            f"distance vocabulary — make sure the tokenizer is co-"
            f"located with the model (see document_structures/AGENTS.md)."
        )
    if len(set(ids)) != _NUM_DISTANCE_BINS:
        raise ValueError(
            f"Tokenizer collapsed distance-bin tokens to {len(set(ids))} "
            f"unique IDs (expected {_NUM_DISTANCE_BINS}). Some <d_X.X> "
            f"tokens are missing from the tokenizer vocab."
        )
    return ids


def _encode_token_strs(tokenizer, token_strs: list[str]) -> list[int]:
    """1:1 encode for whitespace-separated `<...>` tokens (WordLevel)."""
    ids = tokenizer.encode(" ".join(token_strs), add_special_tokens=False)
    if len(ids) != len(token_strs):
        raise ValueError(
            "Tokenizer did not produce 1:1 mapping. "
            f"first 10 in: {token_strs[:10]} / out: {ids[:10]}"
        )
    return [int(x) for x in ids]


# --------------------------------------------------------------------------
# Per-structure inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _StructurePrediction:
    """One inference pass over one (structure, seed-count) — bin probs per pair."""

    entry_id: str
    n_residues: int
    n_seeded: int
    pairs: list[tuple[int, int]]  # (i, j), 1-indexed, i < j
    expected_distances: list[float]   # per pair (Å)
    # 64-bin distribution per pair, in case the caller wants the
    # full thing (e.g. for argmax-based metrics or visualization).
    bin_probs: list[list[float]] | None  # None when we don't keep it


def _query_pairs(
    structure: ParsedStructure,
    *,
    llm,
    tokenizer,
    query_atom: str,
    n_seeded: int,
    top_k_logprobs: int,
    batch_size: int,
    max_pairs: int | None,
    keep_bin_probs: bool,
    distance_token_ids: list[int],
    bin_midpoints,
    gt=None,
    distance_cap_angstrom: float | None = None,
) -> _StructurePrediction:
    """Single (structure, seed-count) inference pass.

    Returns expected distance per pair. Pairs where the model puts
    zero mass on any of the 64 distance bins within top-K are skipped
    from the output.

    If ``gt`` is provided (an N×N distance matrix from
    :func:`_gt_query_distance_matrix`), pairs are pre-filtered: any
    pair with non-finite GT or GT > ``distance_cap_angstrom`` is
    skipped *before* the LLM forward pass. ``max_pairs`` then caps
    the number of evaluatable pairs, not raw queries — which is what
    ``--max-pairs-per-structure`` is meant to control in eval mode.

    The ``predict`` (no-GT) caller passes ``gt=None`` and queries
    every (i, j).
    """
    import numpy as np
    from vllm import SamplingParams, TokensPrompt

    n = len(structure.residues)
    gt_long = _gt_long_range_contacts(structure)
    seeded = gt_long[:n_seeded] if n_seeded > 0 else []
    base_tokens = _build_base_prompt_tokens(structure, seeded)
    base_ids = _encode_token_strs(tokenizer, base_tokens)
    distance_id_set = set(distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(distance_token_ids)}

    prompts: list = []
    keys: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if gt is not None:
                gt_ij = float(gt[i - 1, j - 1])
                if not math.isfinite(gt_ij):
                    continue
                if distance_cap_angstrom is not None and gt_ij > distance_cap_angstrom:
                    continue
            tail = _pair_tail_tokens(i, j, query_atom, query_atom)
            tail_ids = _encode_token_strs(tokenizer, tail)
            prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail_ids))
            keys.append((i, j))
            if max_pairs is not None and len(prompts) >= max_pairs:
                break
        if max_pairs is not None and len(prompts) >= max_pairs:
            break

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, logprobs=top_k_logprobs, n=1,
    )

    out_pairs: list[tuple[int, int]] = []
    out_expected: list[float] = []
    out_bin_probs: list[list[float]] | None = [] if keep_bin_probs else None

    for chunk_start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[chunk_start : chunk_start + batch_size]
        chunk_keys = keys[chunk_start : chunk_start + batch_size]
        outputs = llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for (i, j), out in zip(chunk_keys, outputs, strict=True):
            lp_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            row = np.zeros(_NUM_DISTANCE_BINS, dtype=np.float32)
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in distance_id_set:
                    row[bin_of[tid]] = float(np.exp(float(lp.logprob)))
            total = float(row.sum())
            if total <= 0:
                continue
            row /= total
            expected = float((row * bin_midpoints).sum())
            out_pairs.append((i, j))
            out_expected.append(expected)
            if out_bin_probs is not None:
                out_bin_probs.append(row.tolist())

    return _StructurePrediction(
        entry_id=structure.entry_id,
        n_residues=n,
        n_seeded=n_seeded,
        pairs=out_pairs,
        expected_distances=out_expected,
        bin_probs=out_bin_probs,
    )


def _make_llm(args: argparse.Namespace):
    """Load vllm + tokenizer + resolve distance token IDs. Imports are lazy."""
    import numpy as np
    from vllm import LLM

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        max_logprobs=max(args.top_k_logprobs, 128),
    )
    tokenizer = llm.get_tokenizer()
    distance_token_ids = _resolve_distance_token_ids(tokenizer)
    bin_midpoints = np.asarray(_DISTANCE_BIN_MIDPOINTS, dtype=np.float32)
    return llm, tokenizer, distance_token_ids, bin_midpoints


# --------------------------------------------------------------------------
# Inference class
# --------------------------------------------------------------------------


class V1Inference:
    """``Inference`` Protocol impl for contacts-and-distances-v1."""

    name = NAME
    context_length = CONTEXT_LENGTH

    def __init__(self) -> None:
        self._tokens = all_domain_tokens()

    def tokens(self) -> list[str]:
        return list(self._tokens)

    # ---- arg registration --------------------------------------------------

    def add_args(
        self, parser: argparse.ArgumentParser, *, subcommand: str
    ) -> None:
        """Register flags for `infer` or `evaluate`.

        Common flags (model, input, atom, seeded contacts, vllm knobs)
        are shared. ``infer`` adds ``--keep-bin-probs``. ``evaluate``
        adds nothing extra — the ground truth is the input itself
        (the parsed structure has the coordinates).
        """
        parser.add_argument(
            "--model", required=True,
            help="HuggingFace model path or local dir. Tokenizer is "
                 "loaded from here too (must be co-located).",
        )
        parser.add_argument(
            "--input", type=Path, required=True,
            help="A single structure file (PDB / mmCIF / .gz) or a "
                 "directory of them. For evaluate, the input IS the "
                 "ground truth — its coordinates are compared against "
                 "the model's predictions.",
        )
        parser.add_argument(
            "--query-atom", default="CA",
            help="Atom name to query for both i and j sides of each "
                 "distance statement. Default CA-CA — the most "
                 "directly-trained signal.",
        )
        parser.add_argument(
            "--seed-n-values", type=_parse_seed_n_values, default=(0,),
            help="Comma-separated seeded-contact counts (e.g. '0,5,20,50'). "
                 "Each value runs the full inference pass with that many GT "
                 "long-range contacts prepended to the prompt as hints. "
                 "Default '0' = zero-shot.",
        )
        parser.add_argument(
            "--top-k-logprobs", type=int, default=128,
            help="Top-K logprobs requested from vLLM. Must cover the 64 "
                 "distance bins (default 128 has plenty of headroom).",
        )
        parser.add_argument(
            "--batch-size", type=int, default=64,
            help="Pairs per vLLM generate() call.",
        )
        parser.add_argument(
            "--dtype", default="bfloat16",
            help="vLLM dtype.",
        )
        parser.add_argument(
            "--gpu-memory-utilization", type=float, default=0.85,
            help="vLLM gpu_memory_utilization.",
        )
        parser.add_argument(
            "--max-pairs-per-structure", type=int, default=None,
            help="Cap pairs per structure (useful for smoke tests). In "
                 "evaluate mode this caps *evaluatable* pairs (pairs with "
                 "finite GT below --distance-cap-angstrom); in infer mode "
                 "it caps all queried pairs since there is no GT.",
        )

        if subcommand == "infer":
            parser.add_argument(
                "--keep-bin-probs", action="store_true",
                help="Include the full 64-bin distribution per pair in "
                     "the output records. Default off (records carry "
                     "only the expected distance).",
            )
        elif subcommand == "evaluate":
            parser.add_argument(
                "--distance-cap-angstrom", type=float, default=32.0,
                help="GT distances above this are masked from MAE. "
                     "Anything above lands in the saturated bin so "
                     "predictions for these pairs aren't meaningful.",
            )

    # ---- predict (no GT) ---------------------------------------------------

    def predict(self, args: argparse.Namespace) -> Iterator[dict]:
        """Yield one record per (structure, n_seeded) pair.

        Each record is a dict with the entry id, the seeded-contact
        count, the queried atom pair, the list of (i, j) pairs, and
        the per-pair expected distances. If ``--keep-bin-probs`` is
        set, the full bin probabilities are included too.
        """
        structures = list(iter_parsed_structures(args.input))
        if not structures:
            return
        llm, tokenizer, distance_token_ids, bin_midpoints = _make_llm(args)
        keep_bin_probs = bool(getattr(args, "keep_bin_probs", False))
        for structure in structures:
            for n_seeded in args.seed_n_values:
                pred = _query_pairs(
                    structure,
                    llm=llm,
                    tokenizer=tokenizer,
                    query_atom=args.query_atom,
                    n_seeded=n_seeded,
                    top_k_logprobs=args.top_k_logprobs,
                    batch_size=args.batch_size,
                    max_pairs=args.max_pairs_per_structure,
                    keep_bin_probs=keep_bin_probs,
                    distance_token_ids=distance_token_ids,
                    bin_midpoints=bin_midpoints,
                )
                record: dict[str, Any] = {
                    "entry_id": pred.entry_id,
                    "n_residues": pred.n_residues,
                    "n_seeded": pred.n_seeded,
                    "query_atom": args.query_atom,
                    "pairs": pred.pairs,
                    "expected_distances": pred.expected_distances,
                }
                if pred.bin_probs is not None:
                    record["bin_probs"] = pred.bin_probs
                yield record

    # ---- evaluate (with GT, computes MAE) ----------------------------------

    def evaluate(self, args: argparse.Namespace) -> EvalResult:
        """Run predict + compare against GT distances; return MAE per seed-N.

        Headline metric: ``mae_at_n<N>_angstrom``, macro-mean across
        structures. Per-pair records (entry_id, n_seeded, i, j, gt,
        expected, abs_err) land in ``EvalResult.per_example``.

        Ground truth is the input itself — the GT distance for pair
        (i, j) at atom A is computed from the parsed structure's
        coordinates. Pairs with non-finite GT or GT >
        ``--distance-cap-angstrom`` are masked out.
        """
        import numpy as np

        structures = list(iter_parsed_structures(args.input))
        if not structures:
            return EvalResult(metrics={}, per_example=[], extras={
                "structure": self.name,
                "warning": "no input structures",
                "model": args.model,
            })
        llm, tokenizer, distance_token_ids, bin_midpoints = _make_llm(args)

        # Per (n_seeded, entry_id) -> mae and n_pairs.
        per_structure_mae: dict[int, dict[str, float]] = {
            n: {} for n in args.seed_n_values
        }
        per_structure_n_pairs: dict[int, dict[str, int]] = {
            n: {} for n in args.seed_n_values
        }
        per_pair: list[dict] = []

        for structure in structures:
            gt = _gt_query_distance_matrix(structure, args.query_atom)
            for n_seeded in args.seed_n_values:
                pred = _query_pairs(
                    structure,
                    llm=llm,
                    tokenizer=tokenizer,
                    query_atom=args.query_atom,
                    n_seeded=n_seeded,
                    top_k_logprobs=args.top_k_logprobs,
                    batch_size=args.batch_size,
                    max_pairs=args.max_pairs_per_structure,
                    keep_bin_probs=False,
                    distance_token_ids=distance_token_ids,
                    bin_midpoints=bin_midpoints,
                    gt=gt,
                    distance_cap_angstrom=args.distance_cap_angstrom,
                )
                abs_errs: list[float] = []
                for (i, j), expected in zip(pred.pairs, pred.expected_distances, strict=True):
                    gt_ij = float(gt[i - 1, j - 1])
                    abs_err = abs(expected - gt_ij)
                    abs_errs.append(abs_err)
                    per_pair.append({
                        "entry_id": structure.entry_id,
                        "n_seeded": n_seeded,
                        "i": i,
                        "j": j,
                        "gt_angstrom": gt_ij,
                        "expected_angstrom": expected,
                        "abs_err_angstrom": abs_err,
                    })
                mae = float(np.mean(abs_errs)) if abs_errs else float("nan")
                per_structure_mae[n_seeded][structure.entry_id] = mae
                per_structure_n_pairs[n_seeded][structure.entry_id] = len(abs_errs)

        metrics: dict[str, float] = {}
        for n_seeded in args.seed_n_values:
            maes = [v for v in per_structure_mae[n_seeded].values() if math.isfinite(v)]
            metrics[f"mae_at_n{n_seeded}_angstrom"] = (
                float(np.mean(maes)) if maes else float("nan")
            )

        return EvalResult(
            metrics=metrics,
            per_example=per_pair,
            extras={
                "structure": self.name,
                "model": args.model,
                "query_atom": args.query_atom,
                "seed_n_values": list(args.seed_n_values),
                "n_structures": len(structures),
                "per_structure_mae": per_structure_mae,
                "per_structure_n_pairs": per_structure_n_pairs,
            },
        )


def get_inference() -> V1Inference:
    """Entry point read by the marinfold-document-structure CLI."""
    return V1Inference()
