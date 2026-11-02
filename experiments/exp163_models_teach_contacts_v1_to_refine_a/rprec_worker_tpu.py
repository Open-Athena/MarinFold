# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced **R-precision** on the marin iris TPU pool, via vLLM logprobs.

``eval_refiner_worker.py`` computes the same metric with HF transformers on CUDA. That
path needs a GPU, and neither is available: the CoreWeave batch band has produced nothing
for days and the workstation A5000 is held by another process. vLLM on TPU cannot run a
raw forward pass, but it *can* return the full next-token distribution, which is all the
scorer needs.

The score for a candidate contact (i, j) is the same factorisation exp82/exp89 use::

    score(i, j) = 0.5 * [ (lp1[i] + lp2[i][j]) + (lp1[j] + lp2[j][i]) ]

    lp1[i]     = log P( <p_i> | prefix + <contact> )
    lp2[i][j]  = log P( <p_j> | prefix + <contact> + <p_i> )

so one request gives the whole first factor, and L requests give the second — L+1
single-token requests per protein, which vLLM batches efficiently. The vocabulary is only
2,845, so asking for the full distribution is cheap.

The metric functions (`true_matrix`, `resolved_pairs`, `band_metrics`, `_auc`) are copied
verbatim from ``eval_refiner_worker.py`` so the numbers are directly comparable to every
R-precision figure in this experiment.

**Validation:** the base model's ``R0_all`` on this eval set is a known 0.3355 (measured
independently on CUDA). Running ``--model <E8>`` must reproduce it; if it does not, this
implementation is wrong and its numbers should not be believed.

    python rprec_worker_tpu.py \\
        --model gs://.../hf_bf16/step-35679 \\
        --targets gs://.../eval554/targets.parquet \\
        --prompts gs://.../eval554/prompts \\
        --out gs://.../eval554/rprec/base --shard 0/1 --tpu-type v5p-8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

MIN_SEP = 6
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}


# --------------------------------------------------------------------------
# metrics — copied verbatim from eval_refiner_worker.py (exp89 compute_metrics)
# --------------------------------------------------------------------------
def true_matrix(L: int, pairs) -> np.ndarray:
    m = np.zeros((L, L), bool)
    for i, j in pairs:
        if 0 <= i < j < L:
            m[i, j] = True
    return m


def resolved_pairs(L: int):
    a, b = np.triu_indices(L, k=1)
    return a, b, (b - a)


def _auc(g: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC — identical to sklearn.roc_auc_score, no sklearn."""
    n_pos = int(g.sum())
    n_neg = int(g.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    sv = s[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[g == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def band_metrics(score: np.ndarray, tmat: np.ndarray, pi, pj, psep) -> dict:
    """{band: (R-precision, AUC)} over the upper-triangular candidate universe."""
    cs, cg = score[pi, pj], tmat[pi, pj].astype(int)
    out = {}
    for rng, (lo, hi) in RANGES.items():
        inr = psep >= lo
        if hi is not None:
            inr = inr & (psep <= hi)
        s, g = cs[inr], cg[inr]
        nc, nt = int(s.size), int(g.sum())
        if nc == 0 or nt == 0:
            out[rng] = (float("nan"), float("nan"))
            continue
        order = np.argsort(-s, kind="mergesort")
        top = min(nt, nc)
        out[rng] = (float(g[order][:top].sum()) / top, _auc(g, s))
    return out


def stage(model_path: str) -> str:
    """vLLM needs local files; gs://|s3:// dirs are pulled to tmp."""
    if "://" not in model_path:
        return model_path
    fs, root = url_to_fs(model_path.rstrip("/"))
    local = os.path.join(tempfile.gettempdir(), "rprec_model",
                         hashlib.sha256(model_path.encode()).hexdigest()[:12])
    if not os.path.isdir(local):
        os.makedirs(local, exist_ok=True)
        for f in fs.ls(root):
            name = f.rsplit("/", 1)[-1]
            if name:
                fs.get(f, os.path.join(local, name))
    return local


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None,
                    help="skip proteins longer than this (L+1 requests each)")
    ap.add_argument("--mode-id", type=int, default=None,
                    help="overwrite the doc-type sentinel at position 0 (7 = multi mode)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--tpu-type", default="v5p-8")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    si, sm = (int(x) for x in a.shard.split("/"))
    with fsspec.open(a.targets, "rb") as fh:
        targets = {t["entry_id"]: t for t in pq.read_table(fh).to_pylist()}
    stems = sorted(targets)
    if a.max_len:
        stems = [e for e in stems if int(targets[e]["L"]) <= a.max_len]
    mine = [e for n, e in enumerate(stems) if n % sm == si]
    if a.limit:
        mine = mine[: a.limit]
    print(f"[rprec] shard {si}/{sm}: {len(mine)}/{len(stems)} proteins, "
          f"mode_id={a.mode_id}", flush=True)
    if not mine:
        print(f"[rprec] SHARD_DONE {si}/{sm}: nothing to do", flush=True)
        return

    t0 = time.time()
    local = stage(a.model)
    # Size the logprob request from the model's OWN vocab. Asking for more than the
    # (padded) logit width dies inside the TPU sampler with
    #   ValueError: k argument to top_k must be no larger than size along axis
    # -- the contacts-v1 vocab is 2845 and the padded width 2848, so a naive large
    # constant like 4096 kills the engine after a clean load.
    with open(os.path.join(local, "config.json")) as fh:
        vocab = int(json.load(fh)["vocab_size"])
    llm = LLM(model=local, max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size, enforce_eager=True,
              dtype="bfloat16", max_logprobs=vocab)
    tok = llm.get_tokenizer()
    contact_id = tok.convert_tokens_to_ids("<contact>")
    # Effectively the full distribution (vocab is tiny), so every position token is
    # present and none falls back to the -1e30 sentinel.
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=vocab)
    print(f"[rprec] model loaded in {time.time()-t0:.0f}s "
          f"(contact_id={contact_id}, vocab={vocab})", flush=True)

    pfs, proot = url_to_fs(a.prompts)
    rows = []
    t_all = time.time()
    for n, eid in enumerate(mine):
        t = targets[eid]
        L = int(t["L"])
        gt = [(int(i), int(j)) for i, j in t["gt_contacts"]
              if 0 <= int(i) < L and 0 <= int(j) < L]
        if len(gt) < 5:
            continue
        with pfs.open(f"{proot.rstrip('/')}/{eid}.parquet", "rb") as fh:
            prow = pq.read_table(fh).to_pylist()[0]
        prefix, seq_pos = prow["prefix"], [int(x) for x in prow["seq_positions"]]
        if len(seq_pos) != L:
            print(f"  !! {eid}: prompt L={len(seq_pos)} != {L}, skipping", flush=True)
            continue

        prefix_ids = tok(prefix, add_special_tokens=False).input_ids
        if a.mode_id is not None:
            prefix_ids = [a.mode_id] + list(prefix_ids[1:])
        pos_ids = [tok.convert_tokens_to_ids(f"<p{p}>") for p in seq_pos]
        base = list(prefix_ids) + [contact_id]

        # one request for lp1, then L for lp2 — issued together so vLLM batches them
        prompts = [TokensPrompt(prompt_token_ids=base)]
        prompts += [TokensPrompt(prompt_token_ids=base + [pid]) for pid in pos_ids]
        outs = llm.generate(prompts, sp, use_tqdm=False)

        def dist(o):
            """next-token logprobs over the position tokens, as a dense vector"""
            lp = o.outputs[0].logprobs[0]
            return np.array([lp[t].logprob if t in lp else -1e30 for t in pos_ids],
                            dtype=np.float32)

        lp1 = dist(outs[0])                                  # [L]
        lp2 = np.stack([dist(o) for o in outs[1:]])          # [L, L]
        fwd = lp1[:, None] + lp2
        score = (0.5 * (fwd + fwd.T)).astype(np.float32)

        tmat = true_matrix(L, gt)
        pi, pj, psep = resolved_pairs(L)
        m = band_metrics(score, tmat, pi, pj, psep)
        row = dict(entry_id=eid, L=L, n_gt=len(gt))
        for b in ("all", "short", "medium", "long"):
            row[f"R0_{b}"], row[f"A0_{b}"] = m[b]
        rows.append(row)

        if (n + 1) % 10 == 0 or n == 0:
            el = time.time() - t_all
            print(f"  [{n+1}/{len(mine)}] {eid} L={L} R0_all={row['R0_all']:.3f} "
                  f"({el/(n+1):.1f}s/protein)", flush=True)

    dest = f"{a.out.rstrip('/')}/shard-{si}-of-{sm}.parquet"
    with fsspec.open(dest, "wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows), fh)
    r = np.array([x["R0_all"] for x in rows], float)
    print(f"[rprec] SHARD_DONE {si}/{sm}: {len(rows)} proteins, "
          f"mean R0_all={np.nanmean(r):.4f} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
