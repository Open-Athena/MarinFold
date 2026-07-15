# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp102 rollout worker (local GPU, HF transformers).

For each exp102 target, generate ``--n-rollouts`` contacts-v1 rollouts from the
tuned 1.5B model with the exp82 *resample* recipe (one sample per fresh document
realization), and save — per rollout — everything issue #102's accuracy-factor
analysis needs, which the exp98 worker threw away:

  * predicted contacts in **emission order** (``pred`` = flattened ``[i0,j0,…]``,
    NOT sorted), and
  * each contact's **emission logprob** (``pred_logprob[k]`` = sum of the 3
    sampled-token logprobs of contact k's ``<contact> <pI> <pJ>`` statement,
    from the model's *raw* next-token distribution — ``output_logits``, so top_p/
    top_k warping does not distort the confidence).

Everything else (per-band precision/recall/F1, ``nll``, ``nll_per_tok``,
``n_gen_tokens``, ``finished``) mirrors exp98 so the output joins 1:1 to exp98's
``rollout_metrics_all.parquet`` on ``entry_id`` + ``r``.

Why transformers, not vLLM: this is a single-GPU subset run, and HF ``generate``
gives exact raw per-token logprobs with no vLLM/TPU bf16/rope/tokenizer gotchas.

Writes ``--out/rollout_metrics_ordered/<entry_id>.parquet`` (one row/rollout) and
``--out/timings/<entry_id>.csv``; resumes by skipping targets already written.

    uv run python gen_rollouts_worker_hf.py \
        --model data/model --targets data/targets.parquet --prompts data/prompts \
        --out data/runs/pilot --n-rollouts 1000 --limit 20
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time
import urllib.request

# reduce allocator fragmentation so large-batch generation doesn't OOM on the
# reserved-but-unallocated slack (must be set before torch initializes CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from rollout_metrics import (
    BANDS,
    gt_by_band,
    parse_contacts_ordered,
    parse_pred,
    score_rollout,
)

# Files that make up the HF export on the open-athena bucket.
CKPT_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
DEFAULT_CKPT_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679"
)


def stage_model(dest: str, url: str = DEFAULT_CKPT_URL) -> str:
    """Download the HF export from the bucket to a local dir (idempotent)."""
    os.makedirs(dest, exist_ok=True)
    for name in CKPT_FILES:
        lp = os.path.join(dest, name)
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            continue
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{url}/{name}", lp)
    return dest


def batch_size_for(prefix_len: int, max_new: int) -> int:
    """Rollouts per generate() call, shrunk for long sequences so the KV cache +
    accumulated output_logits (gen_len x [B, vocab]) fit a 24GB card. Tuned for
    SDPA attention (measured: L=273 => bs 128 ~11GB without logits)."""
    total = prefix_len + max_new
    if total <= 768:
        return 192
    if total <= 1536:
        return 112
    if total <= 2304:
        return 72
    if total <= 3072:
        return 48
    return 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/model", help="local model dir (staged if missing)")
    ap.add_argument("--model-url", default=DEFAULT_CKPT_URL)
    ap.add_argument("--targets", default="data/targets.parquet")
    ap.add_argument("--prompts", default="data/prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rollouts", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="first N targets (by entry_id)")
    ap.add_argument("--batch-size", type=int, default=None, help="override adaptive batch")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=102)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(f"{a.out}/rollout_metrics_ordered", exist_ok=True)
    os.makedirs(f"{a.out}/timings", exist_ok=True)

    model_dir = stage_model(a.model, a.model_url)
    tok = AutoTokenizer.from_pretrained(model_dir)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    end_id = tok.convert_tokens_to_ids("<end>")
    print(f"model loaded in {time.time()-t_load:.0f}s (end_id={end_id}, "
          f"vocab={model.config.vocab_size})", flush=True)

    targets = {t["entry_id"]: t for t in pq.read_table(a.targets).to_pylist()}
    stems = sorted(targets)
    if a.limit:
        stems = stems[: a.limit]

    def done(entry):
        return os.path.exists(f"{a.out}/rollout_metrics_ordered/{entry}.parquet")

    todo = stems if a.overwrite else [e for e in stems if not done(e)]
    print(f"{len(stems)} targets assigned, {len(stems)-len(todo)} done, {len(todo)} to do",
          flush=True)

    t_all = time.time()
    for n, entry in enumerate(todo):
        t = targets[entry]
        L = int(t["L"])
        gt = {(int(i), int(j)) for i, j in t["gt_contacts"]}
        gtb = gt_by_band(gt)

        prows = pq.read_table(f"{a.prompts}/{entry}.parquet").to_pylist()[: a.n_rollouts]
        prefixes = [p["prefix"] for p in prows]
        rkeys = [int(p["r"]) for p in prows]
        maps = [{int(pos): i for i, pos in enumerate(p["seq_positions"])} for p in prows]

        # all realizations of a target share prefix token length -> no padding.
        enc = [tok(p, add_special_tokens=False).input_ids for p in prefixes]
        plen = len(enc[0])
        assert all(len(e) == plen for e in enc), f"{entry}: prefixes differ in length"
        max_new = min(a.max_model_len - plen, 4 * L + 64)
        bs = a.batch_size or batch_size_for(plen, max_new)

        rows = []
        total_gen = 0
        t0 = time.time()
        torch.manual_seed(a.seed)
        for s in range(0, len(enc), bs):
            ids = torch.tensor(enc[s:s + bs], device="cuda")
            # all realizations of a target share prefix length -> no left-padding,
            # so the mask is all-ones; pass it explicitly (correctness + silences
            # the "pad == eos" warning).
            attn = torch.ones_like(ids)
            with torch.no_grad():
                out = model.generate(
                    ids, attention_mask=attn,
                    do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                    max_new_tokens=max_new, eos_token_id=end_id, pad_token_id=end_id,
                    return_dict_in_generate=True, output_logits=True,
                )
            seqs = out.sequences[:, plen:]                       # [B, gen_len]
            # raw per-step logprob of the sampled token: log_softmax over the
            # UNwarped logits (output_logits), gathered at the realized token.
            logits = torch.stack(out.logits, dim=1)             # [B, gen_len, vocab]
            logp = torch.log_softmax(logits.float(), dim=-1)
            tok_logp = logp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)  # [B, gen_len]
            del logits, logp

            for bi in range(seqs.shape[0]):
                gi = s + bi
                gen_ids = seqs[bi].tolist()
                # trim at first <end> (everything after is pad/eos continuation).
                if end_id in gen_ids:
                    cut = gen_ids.index(end_id) + 1
                    finished = True
                else:
                    cut = len(gen_ids)
                    finished = False
                gen_ids = gen_ids[:cut]
                ntok = len(gen_ids)
                total_gen += ntok
                step_logp = tok_logp[bi, :ntok]
                nll = float(-step_logp.sum().item())

                token_strs = tok.convert_ids_to_tokens(gen_ids)
                ordered = parse_contacts_ordered(token_strs, maps[gi])  # [(i,j,k),…]
                # per-contact emission logprob = sum of its 3 statement tokens.
                pred_flat, pred_logprob = [], []
                for i, j, k in ordered:
                    pred_flat.extend((i, j))
                    lp = float(step_logp[k:k + 3].sum().item()) if k + 2 < ntok else float("nan")
                    pred_logprob.append(lp)

                # scoring path: identical to exp98 (set-based, from decoded text).
                text = tok.decode(gen_ids, skip_special_tokens=False)
                sc = score_rollout(parse_pred(text, maps[gi]), gtb)

                rows.append(dict(
                    r=rkeys[gi], n_gen_tokens=ntok, finished=finished,
                    n_pred=len(ordered), nll=nll,
                    nll_per_tok=(nll / ntok if ntok else float("nan")),
                    pred=pred_flat, pred_logprob=pred_logprob, **sc,
                ))
            del out, seqs, tok_logp
        gen_s = time.time() - t0

        pq.write_table(pa.Table.from_pylist(rows),
                       f"{a.out}/rollout_metrics_ordered/{entry}.parquet")

        tps = total_gen / gen_s if gen_s > 0 else 0.0
        trow = dict(entry_id=entry, L=L, n_gt=len(gt), n_rollouts=len(rows),
                    plen=plen, max_new=max_new, batch_size=bs,
                    total_gen_tokens=total_gen, gen_seconds=round(gen_s, 2),
                    tokens_per_s=round(tps, 1),
                    mean_gen_tokens=round(total_gen / len(rows), 1),
                    frac_finished=round(sum(r["finished"] for r in rows) / len(rows), 3))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(trow.keys()))
        w.writeheader(); w.writerow(trow)
        with open(f"{a.out}/timings/{entry}.csv", "w") as fh:
            fh.write(buf.getvalue())

        mean_f1 = sum(r["all_f1"] for r in rows) / len(rows)
        best_f1 = max(r["all_f1"] for r in rows)
        print(f"  [{n+1}/{len(todo)}] {entry} L={L} n_gt={len(gt)}  {gen_s:.1f}s  "
              f"{tps:.0f} tok/s  mean_f1={mean_f1:.3f} best_f1={best_f1:.3f}  "
              f"finished={trow['frac_finished']}", flush=True)

    wall = time.time() - t_all
    print(f"\nDONE: {len(todo)} targets, wall {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
