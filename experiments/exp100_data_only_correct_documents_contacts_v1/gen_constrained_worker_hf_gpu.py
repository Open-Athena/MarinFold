"""Local-GPU only-correct rollout worker (exp100) — HF transformers backend.

Functionally identical to gen_constrained_worker_vllm_tpu.py (same only-correct
grammar, same folded-in unmodified-NLL capture, same outputs) but runs the
model with plain HF transformers on a CUDA GPU. This is the **proven** path
(validated end-to-end against a teacher-forced forward pass; see the Phase-0
spike) and the one used for the exp100 validation run while the iris-TPU
logits_processor path is still being probed.

Key efficiency: all N rollouts of a target share the **same** prompt length
(same protein → same residue count → same number of sequence statements) and the
**same** generated length (exactly ``3*n_gt + 1`` tokens under the mask), so a
target's N rollouts batch with zero padding and finish on the same step — one
decode loop per target, no raggedness.

Outputs (mirror the vLLM worker) under ``--out``:
  nll/<entry_id>.parquet        one row per rollout (struct NLL, correctness check)
  documents/<entry_id>.json     the selected (lowest struct-NLL) document
  all_documents/<entry_id>.json all N rollout documents verbatim
  timings/<entry_id>.csv        per-target timing

  python gen_constrained_worker_hf_gpu.py --model <local hf dir> \
      --targets data/targets.parquet --prompts <prompts dir> \
      --out <run dir> --shard 0/1 --n-rollouts 10
"""
import argparse
import csv
import io
import json
import math
import os
import time

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from constrained_grammar import ContactConstraint
from rollout_metrics import gt_by_band, parse_pred, score_rollout


def write_parquet(table, dest):
    with fsspec.open(dest, "wb") as fh:
        pq.write_table(table, fh)


def _top_k_top_p(logits, top_k, top_p):
    """In-place-ish top-k then nucleus (top-p) filtering of a [B, V] logit tensor.
    Operates only on the (already only-correct-masked) logits."""
    import torch
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p and 0 < top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sl, dim=-1).cumsum(dim=-1)
        remove = probs - torch.softmax(sl, dim=-1) >= top_p  # keep tokens up to cumulative top_p
        sl = torch.where(remove, torch.full_like(sl, float("-inf")), sl)
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local HF model dir")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True, help="prompts dir (per-target parquet)")
    ap.add_argument("--out", required=True, help="output run dir (gs:// or local)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-rollouts", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N targets of the shard")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    si, sm = (int(x) for x in a.shard.split("/"))
    out = a.out.rstrip("/")
    ofs, _ = fsspec.core.url_to_fs(out)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    contact_id = tok.convert_tokens_to_ids("<contact>")
    end_id = tok.convert_tokens_to_ids("<end>")
    print(f"model loaded in {time.time()-t_load:.0f}s (contact_id={contact_id}, end_id={end_id})",
          flush=True)

    def pos_token_id(pos):
        return tok.convert_tokens_to_ids(f"<p{pos}>")

    with fsspec.open(a.targets, "rb") as fh:
        targets = {t["entry_id"]: t for t in pq.read_table(fh).to_pylist()}
    stems = sorted(targets)
    mine = stems[si::sm]
    if a.limit:
        mine = mine[: a.limit]

    def done(entry):
        return (ofs.exists(f"{out}/nll/{entry}.parquet")
                and ofs.exists(f"{out}/documents/{entry}.json"))

    if not a.overwrite:
        mine = [e for e in mine if not done(e)]
    print(f"shard {si}/{sm}: {len(mine)} targets to do, n_rollouts={a.n_rollouts}", flush=True)

    @torch.no_grad()
    def run_target(prompt_ids_list, gt_pos_ids_list):
        B = len(prompt_ids_list)
        cons = [ContactConstraint(p, g, contact_id=contact_id, end_id=end_id)
                for p, g in zip(prompt_ids_list, gt_pos_ids_list)]
        ids = torch.tensor(prompt_ids_list, device=a.device)  # [B, plen] (equal len)
        outp = model(input_ids=ids, use_cache=True)
        past = outp.past_key_values
        logits = outp.logits[:, -1]  # [B, V]
        gen = [[] for _ in range(B)]
        max_steps = cons[0].max_new_tokens() + 2
        for _ in range(max_steps):
            masked = torch.stack([cons[b](gen[b], logits[b].clone()) for b in range(B)])  # [B,V]
            filt = _top_k_top_p(masked.float(), a.top_k, a.top_p)
            probs = torch.softmax(filt / a.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)  # [B,1]
            for b in range(B):
                gen[b].append(int(nxt[b, 0]))
            if all(g[-1] == end_id for g in gen):
                break
            step = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = step.past_key_values
            logits = step.logits[:, -1]
        for b in range(B):
            cons[b].finalize(gen[b])
        return gen, cons

    t_all = time.time()
    total_tok_all = 0
    for n, entry in enumerate(mine):
        t = targets[entry]
        L = int(t["L"])
        gt_seq = {(int(i), int(j)) for i, j in t["gt_contacts"]}
        gtb = gt_by_band(gt_seq)
        with fsspec.open(f"{a.prompts}/{entry}.parquet", "rb") as fh:
            prows = pq.read_table(fh).to_pylist()[: a.n_rollouts]

        prompt_ids_list, gt_pos_ids_list, meta = [], [], []
        for p in prows:
            prompt_ids = tok(p["prefix"], add_special_tokens=False).input_ids
            seq_positions = list(p["seq_positions"])
            pos_to_seq = {int(pos): i for i, pos in enumerate(seq_positions)}
            gt_pos_ids = [(pos_token_id(seq_positions[i]), pos_token_id(seq_positions[j]))
                          for i, j in gt_seq]
            prompt_ids_list.append(prompt_ids)
            gt_pos_ids_list.append(gt_pos_ids)
            meta.append(dict(r=int(p["r"]), prefix=p["prefix"], pos_to_seq=pos_to_seq))
        # all prompts share length; guard in case a resample ever differs
        if len({len(x) for x in prompt_ids_list}) != 1:
            raise RuntimeError(f"{entry}: prompt lengths differ across rollouts")

        t0 = time.time()
        gen, cons = run_target(prompt_ids_list, gt_pos_ids_list)
        gen_s = time.time() - t0

        rows, docs, total_gen = [], [], 0
        for i in range(len(prows)):
            g = gen[i]
            total_gen += len(g)
            struct_nll = cons[i].struct_nll()
            nstruct = len(cons[i].token_logprobs)
            text = tok.decode(g, skip_special_tokens=False)
            pred = parse_pred(text, meta[i]["pos_to_seq"])
            sc = score_rollout(pred, gtb)
            finished = g[-1] == end_id
            n_contacts = sum(1 for x in g if x == contact_id)
            rows.append(dict(
                r=meta[i]["r"], n_gen_tokens=len(g), n_contacts=n_contacts,
                finished=finished, n_pred=len(pred),
                all_prec=sc["all_prec"], all_rec=sc["all_rec"], all_f1=sc["all_f1"],
                struct_nll=struct_nll, struct_ntok=nstruct,
                struct_nll_per_tok=(struct_nll / nstruct) if nstruct else math.nan))
            docs.append(dict(r=meta[i]["r"], document=meta[i]["prefix"] + " " + text,
                             struct_nll=struct_nll, n_contacts=n_contacts, finished=finished,
                             all_prec=sc["all_prec"], all_rec=sc["all_rec"],
                             pred_contacts=sorted([list(p) for p in pred])))

        best = min(range(len(rows)), key=lambda i: rows[i]["struct_nll"])
        n_correct = sum(1 for r in rows if r["all_prec"] == 1.0 and r["all_rec"] == 1.0)
        total_tok_all += total_gen

        write_parquet(pa.Table.from_pylist(rows), f"{out}/nll/{entry}.parquet")
        sel_meta = dict(entry_id=entry, L=L, n_gt=len(gt_seq),
                        n_rollouts=len(rows), n_correct=n_correct,
                        selected_by="struct_nll", selected=docs[best],
                        sampling=dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k))
        with fsspec.open(f"{out}/documents/{entry}.json", "w") as fh:
            json.dump(sel_meta, fh)
        with fsspec.open(f"{out}/all_documents/{entry}.json", "w") as fh:
            json.dump(dict(entry_id=entry, documents=docs), fh)

        tps = total_gen / gen_s if gen_s > 0 else 0.0
        trow = dict(entry_id=entry, L=L, n_gt=len(gt_seq), n_rollouts=len(rows),
                    n_correct=n_correct, tensor_parallel=1, total_gen_tokens=total_gen,
                    gen_seconds=round(gen_s, 3), tokens_per_s=round(tps, 1),
                    mean_gen_tokens=round(total_gen / len(rows), 1),
                    best_struct_nll=round(rows[best]["struct_nll"], 3), tpu_type="A5000")
        tbuf = io.StringIO()
        tw = csv.DictWriter(tbuf, fieldnames=list(trow.keys()))
        tw.writeheader(); tw.writerow(trow)
        with fsspec.open(f"{out}/timings/{entry}.csv", "w") as fh:
            fh.write(tbuf.getvalue())
        print(f"  [{n+1}/{len(mine)}] {entry} L={L} n_gt={len(gt_seq)}  {gen_s:.1f}s  "
              f"{tps:.0f} tok/s  correct={n_correct}/{len(rows)}  "
              f"best_struct_nll={rows[best]['struct_nll']:.1f}", flush=True)
        if n_correct != len(rows):
            print(f"  WARNING {entry}: {len(rows)-n_correct} rollouts not 100%-correct", flush=True)

    wall = time.time() - t_all
    print(f"\nSHARD_DONE {si}/{sm}: {len(mine)} targets, {total_tok_all} gen tokens, "
          f"wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
