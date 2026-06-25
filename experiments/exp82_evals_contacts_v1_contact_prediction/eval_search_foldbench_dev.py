# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Inference-algorithm search on the strong contacts-v1 model — FoldBench dev set.

Runs exp82's three rankers — **pairwise / rollout / iterative** (+ a random
baseline) — on the 16-protein FoldBench dev subset (``data/foldbench_dev.jsonl``,
built by ``prepare_foldbench_dev.py``) using the #61/#75 tuned model
(``prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084`` step-35679, eval loss 2.7566).

This is the **dev arm** of the search: exp82 found structured inference didn't
help the *weak* #67 model and concluded "the bottleneck is the base model, not
the readout". With the strong model now in hand (cf. exp27's +30% from iteration
on a strong base model) we re-ask: does rollout/iterative beat pairwise? We
decide that **here**, on FoldBench — not on the held-out contacts-v1 test split.

Prefixes are built with the official, deterministic ``build_document`` (empty
contacts) exactly as exp89's ``score_eval_set``; the candidate universe is each
protein's **resolved** residues and ground truth is the pyconfind contacts
(sep>=6) from the eval-set GT universe.

Run (needs ``marinfold`` on the path)::

    PYTHONPATH=<repo>/marinfold uv run python eval_search_foldbench_dev.py \
        --model /home/bizon/exp89_export/hf_step35679 \
        --methods pairwise,rollout,iterative --n-rollouts 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_contact_prediction import (
    BEGIN,
    MIN_SEP,
    NUM_POS,
    Protein,
    Scorer,
    aggregate,
    candidate_pairs,
    metrics,
    rank_iterative,
    rank_pairwise,
    rank_rollout,
)
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

DEV_JSONL = Path(__file__).parent / "data" / "foldbench_dev.jsonl"


def load_dev_proteins(path: Path) -> list[Protein]:
    """Build a Protein per dev record: full-sequence prefix, resolved candidate
    universe, pyconfind GT — mirroring exp89's prefix construction exactly."""
    proteins: list[Protein] = []
    for line in path.open():
        d = json.loads(line)
        residues = residues_from_sequence(d["input_seq"])
        result = build_document(d["stem"], residues, [], config=GenerationConfig())
        if result is None:
            print(f"  {d['stem']}: build_document returned None; skipping", flush=True)
            continue
        L = result.seq_len
        nterm = result.n_term_index
        seq_positions = [(nterm + k) % NUM_POS for k in range(L)]
        doc = result.document
        prefix = doc[: doc.index(BEGIN) + len(BEGIN)]
        resolved = frozenset(d["resolved"])
        gt = {frozenset((i, j)) for (i, j) in d["contacts"]
              if abs(i - j) >= MIN_SEP and i in resolved and j in resolved}
        proteins.append(Protein(d["stem"], prefix, L, nterm, seq_positions, gt, [], resolved))
    return proteins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", default="pairwise,rollout,iterative")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", type=Path, default=DEV_JSONL)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = args.methods.split(",")
    proteins = load_dev_proteins(args.dev)
    scorer = Scorer(args.model)
    print(f"{len(proteins)} FoldBench dev proteins (resolved universe, pyconfind GT); "
          f"methods={methods}\n", flush=True)

    results: dict[str, list] = {m: [] for m in [*methods, "random"]}
    for p in proteins:
        rng = np.random.default_rng(args.seed)
        rankings: dict[str, list] = {}
        try:
            if "pairwise" in methods:
                rankings["pairwise"] = rank_pairwise(scorer, p)
            if "rollout" in methods:
                rankings["rollout"] = rank_rollout(scorer, p, args.n_rollouts,
                                                   args.temperature, args.top_p)
            if "iterative" in methods:
                rankings["iterative"] = rank_iterative(scorer, p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{p.entry:>9} L={p.L:>3} -- SKIPPED (CUDA OOM)", flush=True)
            continue
        rand = candidate_pairs(p.L, p.resolved)
        rng.shuffle(rand)
        rankings["random"] = rand
        line = [f"{p.entry:>9} L={p.L:>3} res={len(p.resolved):>3} gt={len(p.gt):>3}"]
        for m, rk in rankings.items():
            mm = metrics(rk, p.gt, p.L)
            mm["entry"] = p.entry
            results[m].append(mm)
            line.append(f"{m}:lP@L={mm['long_P@L']:.3f}" if not np.isnan(mm["long_P@L"])
                        else f"{m}:lP@L=n/a")
        print("  ".join(line), flush=True)
        torch.cuda.empty_cache()

    print(f"\n=== AGGREGATE (mean over {len(proteins)} FoldBench dev proteins) ===")
    cols = ["long_P@L", "long_P@L2", "long_P@L5", "long_R",
            "medlong_P@L", "medlong_R", "P@ngt"]
    print(f"{'method':<12} " + " ".join(f"{c:>11}" for c in cols))
    for m in [*methods, "random"]:
        if not results[m]:
            continue
        a = aggregate(results[m])
        print(f"{m:<12} " + " ".join(f"{a.get(c, float('nan')):>11.3f}" for c in cols))
    print("\n(long = seq-sep>=24, medlong = >=12; P@L/L2/L5 = precision of top-{L,L/2,L/5}; "
          "R = R-precision = precision at top-#GT in band.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
