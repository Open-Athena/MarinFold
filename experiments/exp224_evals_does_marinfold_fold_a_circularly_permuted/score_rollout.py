#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step B — exp82 rollout+resample contact scoring for the exp224 units, via CUDA vLLM.

This is exp82's ``score_rollout_vllm.py`` with one change: it reads the four
exp224 units from ``data/units.json`` instead of the 554-protein eval set. The
sampling recipe is *fixed* and must not drift, because every published MarinFold
contact number is on it:

* 100 rollouts per protein, each from a **fresh** document realization
  (resampled N-terminus + ``<pX> <AA>`` statement order),
* temperature 1.0, top-p 0.95, **top-k disabled** (``-1``),
* token budget ``6L+128``,
* vote by per-pair occurrence frequency, no pairwise tie-break.

Because exp224 has 4 proteins rather than 554, we can afford to repeat the whole
thing under several independent seeds. That is the point of ``--seed``: rollout
scoring is stochastic, and a CP-vs-WT gap on n=1 protein per arm is worthless
without knowing the seed-to-seed spread. Each seed writes its own directory.

Self-contained on purpose (no exp82 import chain): the vLLM venv holds vllm +
transformers + numpy + marinfold(--no-deps) and nothing else.

    .venv-vllm/bin/python score_rollout.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
HERE = Path(__file__).resolve().parent


def realization(stem: str, residues, tag: str):
    """One contacts-v1 realization: prefix string + per-seq-index position list."""
    res = build_document(f"{stem}:{tag}", residues, [], config=GenerationConfig())
    seq_positions = [(res.n_term_index + k) % NUM_POS for k in range(res.seq_len)]
    prefix = res.document[: res.document.index(BEGIN) + len(BEGIN)]
    return prefix, seq_positions


def vote_matrix(texts, maps, L):
    """[L,L] symmetric per-pair occurrence frequency; each text parsed with ITS map."""
    M = np.zeros((L, L), np.float32)
    n_contacts = 0
    for text, seqidx in zip(texts, maps):
        seen = set()
        for a, b in CONTACT_RE.findall(text):
            ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if abs(ia - ib) >= MIN_SEP and key not in seen:
                seen.add(key)
                M[key[0], key[1]] += 1
                M[key[1], key[0]] += 1
        n_contacts += len(seen)
    return M, n_contacts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="local HF dir; default = resolve MODELS.yaml's default entry")
    ap.add_argument("--units", type=Path, default=HERE / "data" / "units.json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default _scratch/scores/seed<N>")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 = disabled (vLLM convention)")
    ap.add_argument("--contact-mult", type=int, default=6, help="budget = mult*L + 128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--save-rollouts", action="store_true",
                    help="also dump raw completions (for the notebook / debugging)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    model = args.model
    if model is None:
        from marinfold.registry import resolve_model, resolve_model_entry
        os.environ.setdefault("MARINFOLD_MODELS_YAML",
                              str(HERE.parent.parent / "marinfold" / "marinfold" / "MODELS.yaml"))
        entry = resolve_model_entry(None)
        model = str(resolve_model(None))
        print(f"resolved default model -> {entry.nickname} @ {model}")

    out_dir = args.out_dir or (HERE / "_scratch" / "scores" / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    units = json.loads(args.units.read_text())
    records = sorted(units.values(), key=lambda r: r["L"])

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=args.gpu_frac,
              max_num_seqs=args.max_num_seqs, seed=args.seed, enforce_eager=False)

    meta = []
    t0 = time.time()
    for rec in records:
        L, stem, unit = rec["L"], rec["pdb"].lower(), rec["unit"]
        residues = residues_from_sequence(rec["input_seq"])
        prompts, maps = [], []
        for k in range(args.n_rollouts):
            # Distinct tag per (seed, rollout) so seeds do not share realizations.
            prefix, seq_positions = realization(stem, residues, f"s{args.seed}r{k}")
            prompts.append(prefix)
            maps.append({p: i for i, p in enumerate(seq_positions)})
        budget = args.contact_mult * L + 128
        sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_tokens=budget, seed=None)
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
        finished = sum(1 for o in outs if o.outputs[0].finish_reason == "stop")
        M, n_contacts = vote_matrix(texts, maps, L)
        np.savez_compressed(out_dir / f"{unit}.npz", score=M.astype(np.float16))
        if args.save_rollouts:
            (out_dir / f"{unit}.rollouts.json").write_text(json.dumps(texts))
        row = dict(unit=unit, pdb=rec["pdb"], L=L, n_rollouts=args.n_rollouts,
                   budget=budget, frac_finished=finished / len(outs),
                   mean_contacts_per_rollout=n_contacts / len(outs),
                   seconds=round(time.time() - t1, 1))
        meta.append(row)
        print(f"  {unit:10s} L={L:4d} finished={row['frac_finished']:.3f} "
              f"contacts/rollout={row['mean_contacts_per_rollout']:.1f} "
              f"({row['seconds']}s)")

    prov = dict(
        model=model, seed=args.seed, n_rollouts=args.n_rollouts,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        contact_mult=args.contact_mult, min_sep=MIN_SEP,
        total_seconds=round(time.time() - t0, 1), units=meta,
    )
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
    unfinished = [m["unit"] for m in meta if m["frac_finished"] < 1.0]
    if unfinished:
        print(f"WARNING: rollouts hit the token cap for {unfinished} — scores truncated")
    print(f"\nwrote {out_dir} in {prov['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
