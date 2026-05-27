# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: measure the model's entropy over position tokens given
each contact-range token.

For each protein in the train set, for each of the 3 range tokens,
construct ``<begin_sequence><AAs><begin_statements><{range}-range-contact>``,
ask vLLM for the next-token logprobs at that position, and compute
the entropy of the position-token marginal distribution (positions
restricted to ``<p1>..<pN>``).

Output: data/range_entropy.csv with columns
``stem, n_residues, entropy_long, entropy_med, entropy_short,
top_p_long, top_p_med, top_p_short``.

If the entropy is low and similar across ranges, the model has signal
on all 3. If one range is dramatically flatter (higher entropy), it
suggests the model doesn't really know how to pick positions for that
range and sampling from it adds noise.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from naive_inference import (  # noqa: E402
    Runtime, _build_base_prompt, _encode_tokens, load_runtime,
)
from canonical_sequence import read_canonical_sequence  # noqa: E402
from sampled_contacts_inference import (  # noqa: E402
    _CONTACT_TOKEN_STRS,
    _resolve_position_token_ids,
)


def measure_range_entropies(
    rt: Runtime,
    residue_names: list[str],
) -> dict:
    """Return per-range entropy of next-token distribution over positions.

    Restricted to <p1>..<pN>; renormalised inside that subset.
    """
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    pos_ids = _resolve_position_token_ids(rt.tokenizer, n)
    pos_id_set = set(pos_ids)

    base_tokens = _build_base_prompt(residue_names)
    base_ids = _encode_tokens(rt.tokenizer, base_tokens)

    out: dict[str, dict] = {}
    max_logprobs = 128  # vLLM's default cap. May miss some position tokens
    # for the longer proteins, but the comparison across the 3 ranges stays fair
    # since we use the same cap everywhere.

    prompts = []
    for tok in _CONTACT_TOKEN_STRS:
        tail_id = rt.tokenizer.encode(tok, add_special_tokens=False)
        if len(tail_id) != 1:
            raise ValueError(f"{tok} didn't tokenize to 1")
        prompts.append(TokensPrompt(prompt_token_ids=base_ids + [tail_id[0]]))

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, n=1, logprobs=max_logprobs,
    )
    outputs = rt.llm.generate(prompts, sampling, use_tqdm=False)
    for tok, gen in zip(_CONTACT_TOKEN_STRS, outputs):
        lp_dict = gen.outputs[0].logprobs[0] if gen.outputs[0].logprobs else {}
        pos_probs = np.zeros(n, dtype=np.float64)
        for tid, lp in lp_dict.items():
            tid = int(tid)
            if tid not in pos_id_set:
                continue
            # Find position index (1..N) from token id
            idx = pos_ids.index(tid)  # O(N) but called once per protein
            pos_probs[idx] = math.exp(float(lp.logprob))
        total = pos_probs.sum()
        if total > 0:
            pos_probs /= total
        entropy_bits = -float(
            (pos_probs[pos_probs > 0] * np.log2(pos_probs[pos_probs > 0])).sum()
        )
        top_p = float(pos_probs.max())
        out[tok] = {
            "entropy_bits": entropy_bits,
            "max_entropy_bits": math.log2(n),
            "top_p": top_p,
            "captured_mass": float(total),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv", type=Path,
        default=_THIS / "data" / "train_proteins.csv",
    )
    parser.add_argument(
        "--protenix-dir", type=Path,
        default=_THIS / "protenix_data" / "data" / "protenix-foldbench-monomers",
    )
    parser.add_argument("--model", default="1B")
    parser.add_argument(
        "--models-yaml", type=Path,
        default=_THIS.parent.parent / "marinfold" / "marinfold" / "MODELS.yaml",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--out-csv", type=Path,
        default=_THIS / "data" / "range_entropy.csv",
    )
    args = parser.parse_args()

    rt = load_runtime(model_nickname=args.model, models_yaml=args.models_yaml, dtype=args.dtype)
    rows = list(csv.DictReader(args.train_csv.open()))

    print(f"{'stem':>8}  {'N':>3}  {'H_max':>5}  "
          f"{'H_long':>6}  {'H_med':>6}  {'H_short':>6}  "
          f"{'topP_long':>9}  {'topP_med':>9}  {'topP_short':>9}  "
          f"{'mass_l':>6}  {'mass_m':>6}  {'mass_s':>6}")

    out_rows = []
    for r in rows:
        stem = r["stem"]
        seq = read_canonical_sequence(args.protenix_dir / "gt" / f"{stem}.cif")
        ent = measure_range_entropies(rt, seq.residue_names)
        Hmax = math.log2(seq.n_residues)
        long = ent["<long-range-contact>"]
        med = ent["<medium-range-contact>"]
        short = ent["<short-range-contact>"]
        print(
            f"{stem:>8}  {seq.n_residues:>3}  {Hmax:>5.2f}  "
            f"{long['entropy_bits']:>6.2f}  {med['entropy_bits']:>6.2f}  "
            f"{short['entropy_bits']:>6.2f}  "
            f"{long['top_p']:>9.4f}  {med['top_p']:>9.4f}  {short['top_p']:>9.4f}  "
            f"{long['captured_mass']:>6.3f}  {med['captured_mass']:>6.3f}  "
            f"{short['captured_mass']:>6.3f}"
        )
        out_rows.append({
            "stem": stem,
            "n_residues": seq.n_residues,
            "entropy_max_bits": round(Hmax, 4),
            "entropy_long_bits": round(long["entropy_bits"], 4),
            "entropy_med_bits": round(med["entropy_bits"], 4),
            "entropy_short_bits": round(short["entropy_bits"], 4),
            "top_p_long": round(long["top_p"], 6),
            "top_p_med": round(med["top_p"], 6),
            "top_p_short": round(short["top_p"], 6),
            "captured_mass_long": round(long["captured_mass"], 4),
            "captured_mass_med": round(med["captured_mass"], 4),
            "captured_mass_short": round(short["captured_mass"], 4),
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
