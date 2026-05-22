# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""GT-shell-filtered MarinFold 1B distogram readout.

Same algorithm as ``naive_inference.py``, with one change: instead of
querying the model for every (i, j) pair, we first build a mask from
the GT CB-CB distance matrix and only query pairs where
``gt_d < 15 Å`` (the LDDT inclusion radius from
``score_marinfold._LDDT_INCLUSION_RADIUS_A``). Pairs outside the shell
are left as all-zero probability vectors in the saved distogram —
``score_marinfold._lddt_point`` only averages over the same GT-shell
mask, so point LDDT is unchanged. MAE / dRMSD / contact-precision
columns become biased (they touch out-of-shell pairs) — we treat the
headline ``lddt_distogram_cb`` as the only valid score for this
algorithm.

This is a **GT-aware** speedup: it cannot be run on held-out proteins
without GT structures. Its purpose is to free up the wall-clock budget
for richer inference algorithms (multi-rollout, sampling, …) on the
10-protein train set. Per-protein speedup ≈ N² / #pairs-in-15Å-shell,
which empirically lands around 7× on this train set (15% of pairs in
shell, dominated by the longer proteins).
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
from score_marinfold import (  # noqa: E402
    _LDDT_INCLUSION_RADIUS_A,
    _pairwise_distance_matrix,
    _read_gt_rep_coords,
)


def build_gt_shell_mask(gt_cif: Path, n_residues_seq: int) -> np.ndarray:
    """Return an ``[N, N]`` bool mask of upper-triangle pairs with ``gt_d < 15 Å``.

    Diagonal and lower triangle are False so the readout never duplicates
    work. ``score_marinfold`` symmetrises the predicted probs via
    ``probs[j-1, i-1] = probs[i-1, j-1]`` so leaving the lower triangle
    empty here is fine: the readout fills both halves.
    """
    n_gt, rep = _read_gt_rep_coords(gt_cif)
    if n_gt != n_residues_seq:
        raise ValueError(
            f"GT/seq length mismatch for {gt_cif}: gt={n_gt}, seq={n_residues_seq}"
        )
    d, gt_mask = _pairwise_distance_matrix(rep)
    i_idx = np.arange(n_gt)[:, None]
    j_idx = np.arange(n_gt)[None, :]
    upper = j_idx > i_idx
    return gt_mask & upper & (d < _LDDT_INCLUSION_RADIUS_A)


def predict_distogram_gt_filtered(
    *,
    rt: Runtime,
    residue_names: list[str],
    pair_mask: np.ndarray,
    batch_size: int = 128,
    top_k_logprobs: int = 128,
) -> tuple[np.ndarray, int]:
    """Naive distogram readout, restricted to ``pair_mask`` pairs.

    Returns ``(probs, n_pairs_queried)``. ``probs`` is ``[N, N, 64]``
    with zeros for pairs outside the mask. Symmetrised across the
    diagonal exactly like ``naive_inference.predict_distogram``.
    """
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    if pair_mask.shape != (n, n):
        raise ValueError(f"pair_mask shape {pair_mask.shape} != ({n}, {n})")

    base_tokens = _build_base_prompt(residue_names)
    base_ids = _encode_tokens(rt.tokenizer, base_tokens)
    distance_id_set = set(rt.distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(rt.distance_token_ids)}

    pairs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if pair_mask[i - 1, j - 1]:
                pairs.append((i, j))

    atoms = [
        (representative_atom_name(residue_names[i - 1]),
         representative_atom_name(residue_names[j - 1]))
        for i, j in pairs
    ]

    prompts = []
    for (i, j), (a_i, a_j) in zip(pairs, atoms, strict=True):
        tail = _encode_tokens(rt.tokenizer, [
            "<distance>", f"<p{i}>", f"<p{j}>", f"<{a_i}>", f"<{a_j}>",
        ])
        prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail))

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
    return probs, len(pairs)


def predict_one(
    *,
    rt: Runtime,
    stem: str,
    protenix_dir: Path,
    out_dir: Path,
    batch_size: int = 128,
    algorithm: str = "gt_filtered_naive",
) -> float:
    """Predict the GT-filtered distogram for one stem.

    Same provenance schema as ``naive_inference.predict_one`` with two
    extra fields: ``n_pairs_queried`` and ``pair_filter_fraction``.
    """
    gt_cif = protenix_dir / "gt" / f"{stem}.cif"
    seq = read_canonical_sequence(gt_cif)
    pair_mask = build_gt_shell_mask(gt_cif, seq.n_residues)
    n_pairs_total = seq.n_residues * (seq.n_residues - 1) // 2

    t_start = time.time()
    probs, n_pairs_queried = predict_distogram_gt_filtered(
        rt=rt, residue_names=seq.residue_names, pair_mask=pair_mask,
        batch_size=batch_size,
    )
    elapsed = time.time() - t_start

    out_path = out_dir / stem / "distogram.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, probs=probs)

    (out_path.parent / "provenance.json").write_text(json.dumps({
        "stem": stem,
        "n_residues": seq.n_residues,
        "n_pairs": n_pairs_total,
        "n_pairs_queried": n_pairs_queried,
        "pair_filter_fraction": round(n_pairs_queried / n_pairs_total, 4),
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
    args = parser.parse_args()
    rt = load_runtime(
        model_nickname=args.model, models_yaml=args.models_yaml,
        dtype=args.dtype,
    )
    elapsed = predict_one(
        rt=rt, stem=args.stem,
        protenix_dir=args.protenix_dir, out_dir=args.out,
        batch_size=args.batch_size,
    )
    print(f"{args.stem}: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
