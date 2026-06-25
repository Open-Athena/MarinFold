# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""pairwise vs rollout+resample on the full curated eval set (554 proteins).

The dev-set search settled the inference recipe: **rollout + document-resampling**
(rollout > pairwise; sampling-tuning a wash; resampling a small consistent bonus).
This evaluates it on the whole #74/#78 curated set — FoldBench-100 + 454
denovo_pdb / casp_fm / cameo_hard PDB proteins — alongside a re-run of **pairwise**
(which should reproduce exp89's number and is the apples-to-apples baseline).

Per protein we record metrics for both methods AND wall-clock **timings** (the
rollout+resample time is what we plot vs sequence length). The run is
**resumable**: each protein is appended to the output JSONL as it finishes, and a
restart skips proteins already present. Proteins are processed short→long so most
of the set lands early and the long FoldBench tail is last.

GT = pyconfind contacts (sep>=6) over the resolved residues (exp89 gt_universe);
prefixes via the official build_document. Run (needs marinfold on the path)::

    PYTHONPATH=<repo>/marinfold uv run python eval_full_curated_set.py \
        --model /home/bizon/exp89_export/hf_step35679 \
        --out _scratch/eval_full_results.jsonl --n-rollouts 100
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_contact_prediction import BEGIN, MIN_SEP, NUM_POS, Protein, Scorer, metrics, rank_pairwise
from eval_rollout_resampled_dev import rank_rollout_resampled
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

GT_UNIVERSE = Path(
    "/home/bizon/git/MarinFold/.claude/worktrees/vibrant-hermann-12cd27/"
    "experiments/exp89_evals_contacts_v1_model_on_eval_set/data/gt_universe.jsonl"
)
MANIFESTS = (
    Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/"
         "data/eval_manifest_foldbench.csv"),
    Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/"
         "data/eval_manifest_exp65.csv"),
)


def load_eval_records():
    """Join the GT universe (554 proteins) with the manifest sequences."""
    seqs = {}
    for m in MANIFESTS:
        for r in csv.DictReader(m.open()):
            seqs[(r["dataset"], r["stem"])] = r["input_seq"]
    recs = []
    for line in GT_UNIVERSE.open():
        d = json.loads(line)
        contacts = sorted({(min(int(c[0]), int(c[1])), max(int(c[0]), int(c[1])))
                           for c in d["contacts"] if abs(int(c[0]) - int(c[1])) >= MIN_SEP})
        recs.append(dict(dataset=d["dataset"], stem=d["stem"], L=d["L"],
                         n_resolved=d["n_resolved"], resolved=d["resolved"],
                         contacts=contacts, input_seq=seqs[(d["dataset"], d["stem"])]))
    recs.sort(key=lambda r: r["L"])  # short -> long
    return recs


def prefix_for(entry_id, residues):
    res = build_document(entry_id, residues, [], config=GenerationConfig())
    L = res.seq_len
    seq_positions = [(res.n_term_index + k) % NUM_POS for k in range(L)]
    prefix = res.document[: res.document.index(BEGIN) + len(BEGIN)]
    return prefix, seq_positions, L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="probe: only the first N (by length)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                r = json.loads(line)
                done.add((r["dataset"], r["stem"]))
            except json.JSONDecodeError:
                pass

    records = load_eval_records()
    todo = [r for r in records if (r["dataset"], r["stem"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(records)} eval proteins | {len(done)} done | {len(todo)} to do "
          f"| n_rollouts={args.n_rollouts} batch={args.batch}", flush=True)

    scorer = Scorer(args.model)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    with args.out.open("a") as fout:
        for k, r in enumerate(todo):
            stem, residues = r["stem"], residues_from_sequence(r["input_seq"])
            resolved = frozenset(r["resolved"])
            gt = {frozenset(c) for c in r["contacts"] if c[0] in resolved and c[1] in resolved}
            prefix, seq_positions, L = prefix_for(stem, residues)  # canonical realization (matches exp89)
            p = Protein(stem, prefix, L, seq_positions[0], seq_positions, gt, [], resolved)

            t0 = time.time()
            rk_pw = rank_pairwise(scorer, p)
            t_pw = time.time() - t0

            torch.manual_seed(args.seed)
            t0 = time.time()
            rk_rs = rank_rollout_resampled(scorer, stem, residues, resolved, args.n_rollouts,
                                           args.temperature, args.top_p, args.top_k, args.batch)
            t_rs = time.time() - t0

            rec = dict(dataset=r["dataset"], stem=stem, L=L, n_resolved=r["n_resolved"],
                       n_gt=len(gt), pairwise=metrics(rk_pw, gt, L), resample=metrics(rk_rs, gt, L),
                       t_pairwise_s=round(t_pw, 3), t_resample_s=round(t_rs, 3),
                       n_rollouts=args.n_rollouts, gpu=gpu)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            el = time.time() - t_start
            print(f"  [{k + 1}/{len(todo)}] {stem:>9} L={L:>3}  pw={t_pw:5.1f}s  "
                  f"resample={t_rs:6.1f}s  (elapsed {el / 60:.1f}m)", flush=True)
            torch.cuda.empty_cache()
    print(f"[done] {len(todo)} proteins in {(time.time() - t_start) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
