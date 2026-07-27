# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""rollout+resample contact scoring on the 554-protein eval set, via CUDA vLLM.

Same measurement as exp82's ``score_rollout_resample_eval.py`` — for each eval
protein draw N contacts-v1 rollouts, each from a *fresh* document realization
(resampled N-terminus + statement order), and accumulate per-pair occurrence
frequency into a symmetric ``[L, L]`` float16 matrix in input-sequence
coordinates, saved as ``scores/<dataset>__<stem>.npz`` (key ``score``) so exp89's
``compute_metrics.py`` can score it unchanged.

Two deliberate differences from the exp82 HF-transformers version:

* **vLLM instead of ``model.generate``.** Continuous batching means a rollout
  that emits ``<end>`` early stops costing compute, instead of padding out to the
  batch maximum. ~6x faster end to end on one A5000, which is what makes
  re-running every model under a changed sampling recipe affordable.
* **``top_k`` disabled by default and a 6L+128 token budget.** Top-k renormalises
  over the kept tokens, which inflates ``<end>`` and shortens documents; exp142
  (#142) made that knob the question. Untruncated sampling emits longer
  documents, so the exp82 budget of ``4L+64`` would start truncating them —
  ``6L+128`` is exp142's generous budget (the fullest eval GT document needs
  ~4.2L contact tokens). ``frac_finished`` in the run log proves the cap never
  binds.

Self-contained on purpose (no exp82 import chain): the vLLM venv holds vllm +
transformers + numpy and nothing else.

    python score_rollout_vllm.py --model <hf-dir> --out-dir <scores/> --n-rollouts 100
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/bizon/git/MarinFold/marinfold")
from marinfold.document_structures.contacts_v1 import (  # noqa: E402
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

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
    """Join the exp89 GT universe (554 proteins) with the manifest sequences."""
    seqs = {}
    for m in MANIFESTS:
        for r in csv.DictReader(m.open()):
            seqs[(r["dataset"], r["stem"])] = r["input_seq"]
    recs = []
    for line in GT_UNIVERSE.open():
        d = json.loads(line)
        recs.append(dict(dataset=d["dataset"], stem=d["stem"], L=d["L"],
                         input_seq=seqs[(d["dataset"], d["stem"])]))
    recs.sort(key=lambda r: r["L"])  # short -> long
    return recs


def realization(stem, residues, tag):
    """One contacts-v1 realization: prefix string + per-seq-index position list."""
    res = build_document(f"{stem}:{tag}", residues, [], config=GenerationConfig())
    seq_positions = [(res.n_term_index + k) % NUM_POS for k in range(res.seq_len)]
    prefix = res.document[: res.document.index(BEGIN) + len(BEGIN)]
    return prefix, seq_positions


def vote_matrix(texts, maps, L):
    """[L,L] symmetric per-pair occurrence frequency; each text parsed with ITS map."""
    M = np.zeros((L, L), np.float32)
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
    return M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 = disabled (vLLM convention)")
    ap.add_argument("--contact-mult", type=int, default=6, help="budget = mult*L + 128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=8, help="proteins submitted per generate() call")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--skip", type=int, default=0, help="skip the N shortest (probing only)")
    ap.add_argument("--stride", type=int, default=1,
                    help="score every Nth protein in length order (length-stratified subset)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_eval_records()
    todo = [r for r in records
            if not (args.out_dir / f"{r['dataset']}__{r['stem']}.npz").exists()]
    if args.stride > 1:
        records = records[::args.stride]
        todo = [r for r in records
                if not (args.out_dir / f"{r['dataset']}__{r['stem']}.npz").exists()]
    if args.skip:
        todo = todo[args.skip:]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(records)} eval proteins | {len(records) - len(todo)} already scored | "
          f"{len(todo)} to do | n_rollouts={args.n_rollouts} top_k={args.top_k} "
          f"top_p={args.top_p} T={args.temperature}", flush=True)
    if not todo:
        return 0

    tok = AutoTokenizer.from_pretrained(args.model)
    end_id = tok.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    # generation_config="vllm": ignore the checkpoint's baked-in generation defaults
    # (the exp75 export carries top_k=50) so the recipe comes only from SamplingParams.
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=args.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=args.max_num_seqs, seed=args.seed)

    t0, n_unfinished, n_total = time.time(), 0, 0
    for s in range(0, len(todo), args.chunk):
        group = todo[s:s + args.chunk]
        prompts, per, sps = [], [], []
        for r in group:
            residues = residues_from_sequence(r["input_seq"])
            maps = []
            first = len(prompts)
            for k in range(args.n_rollouts):
                prefix, sp = realization(r["stem"], residues, f"r{k}")
                prompts.append(prefix)
                maps.append({pos: i for i, pos in enumerate(sp)})
            plen = len(tok(prompts[first], add_special_tokens=False).input_ids)
            max_new = min(8192 - plen, args.contact_mult * r["L"] + 128)
            per.append((r, first, maps))
            sps += [SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                   top_k=args.top_k, max_tokens=max_new,
                                   stop_token_ids=[end_id], skip_special_tokens=False,
                                   seed=args.seed * 1_000_003 + first + k)
                    for k in range(args.n_rollouts)]
        ts = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - ts
        for r, first, maps in per:
            chunk_outs = outs[first:first + args.n_rollouts]
            texts = [o.outputs[0].text for o in chunk_outs]
            unfin = sum(1 for o in chunk_outs if o.outputs[0].finish_reason != "stop")
            n_unfinished += unfin
            n_total += args.n_rollouts
            M = vote_matrix(texts, maps, r["L"])
            np.savez_compressed(args.out_dir / f"{r['dataset']}__{r['stem']}.npz",
                                score=M.astype(np.float16))
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        done = s + len(group)
        print(f"  [{done}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
              f"{dt:6.1f}s  {ntok / dt:7.0f} tok/s  unfinished={n_unfinished}/{n_total}  "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)
    print(f"[done] {len(todo)} proteins in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}\n"
          f"[done] unfinished rollouts (hit the token cap): {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
