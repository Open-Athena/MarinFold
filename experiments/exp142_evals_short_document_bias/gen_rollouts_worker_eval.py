# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Short-document-bias probe worker (local GPU, HF transformers).

Generate ``--n-rollouts`` contacts-v1 rollouts per eval target from the tuned 1.5B
model with the exp82 *resample* recipe, and record per rollout exactly what the
"does the model emit far fewer contacts / far shorter documents than ground truth"
question needs:

  * ``n_gen_tokens``     — length of the generated contact section (incl. ``<end>``)
  * ``finished``         — did the rollout emit ``<end>`` before the token budget?
                           (so a short document is never a truncation artifact)
  * ``n_contact_stmts``  — raw count of ``<contact>`` tokens emitted (pre-dedup)
  * ``n_pred``           — deduped, valid, sep>=6 predicted contacts
  * per-band npred/tp/prec/rec/f1 (via exp98/exp102 ``score_rollout``)

Differs from ``gen_rollouts_worker_hf.py`` in two deliberate ways:
  * **No ``output_logits``.** We don't need per-contact logprobs here; dropping
    them frees the memory to raise the token budget and batch size.
  * **Generous budget** ``max_new = min(max_model_len - plen, --contact-mult * L
    + 128)`` (default mult=6). The fullest eval GT document needs ~4.2*L contact
    tokens (n_gt/L maxes at ~1.4, x3 tokens/contact); 6*L+128 leaves comfortable
    headroom so the cap never truncates a genuine full document. ``frac_finished``
    reports whether any rollout still ran to the cap.

    uv run python gen_rollouts_worker_eval.py \
        --targets data/eval/targets.parquet --prompts data/eval/prompts \
        --out data/eval/runs/probe --n-rollouts 200
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from rollout_metrics import gt_by_band, parse_pred, score_rollout

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
    os.makedirs(dest, exist_ok=True)
    for name in CKPT_FILES:
        lp = os.path.join(dest, name)
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            continue
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{url}/{name}", lp)
    return dest


def batch_size_for(total: int) -> int:
    """Rollouts per generate() call by total (prefix+gen) length. No output_logits
    here, so a touch larger than the exp102 worker; tuned for a 24GB card + SDPA."""
    if total <= 768:
        return 224
    if total <= 1536:
        return 144
    if total <= 2304:
        return 96
    if total <= 3072:
        return 64
    if total <= 4608:
        return 40
    return 28


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/model")
    ap.add_argument("--model-url", default=DEFAULT_CKPT_URL)
    ap.add_argument("--targets", default="data/eval/targets.parquet")
    ap.add_argument("--prompts", default="data/eval/prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rollouts", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--contact-mult", type=int, default=6,
                    help="token budget = min(max_model_len-plen, mult*L+128)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=102)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(f"{a.out}/rollout_metrics", exist_ok=True)
    os.makedirs(f"{a.out}/timings", exist_ok=True)

    model_dir = stage_model(a.model, a.model_url)
    tok = AutoTokenizer.from_pretrained(model_dir)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    contact_id = tok.convert_tokens_to_ids("<contact>")
    end_id = tok.convert_tokens_to_ids("<end>")
    print(f"model loaded in {time.time()-t_load:.0f}s (end_id={end_id}, "
          f"contact_id={contact_id}, vocab={model.config.vocab_size})", flush=True)

    targets = {t["entry_id"]: t for t in pq.read_table(a.targets).to_pylist()}
    stems = sorted(targets, key=lambda e: targets[e]["L"])  # short -> long
    if a.limit:
        stems = stems[: a.limit]

    def done(entry):
        return os.path.exists(f"{a.out}/rollout_metrics/{entry}.parquet")

    todo = stems if a.overwrite else [e for e in stems if not done(e)]
    print(f"{len(stems)} targets assigned, {len(stems)-len(todo)} done, {len(todo)} to do",
          flush=True)

    t_all = time.time()
    for n, entry in enumerate(todo):
        t = targets[entry]
        L = int(t["L"])
        n_gt = int(t["n_gt"])
        gt = {(int(i), int(j)) for i, j in t["gt_contacts"]}
        gtb = gt_by_band(gt)

        prows = pq.read_table(f"{a.prompts}/{entry}.parquet").to_pylist()[: a.n_rollouts]
        prefixes = [p["prefix"] for p in prows]
        rkeys = [int(p["r"]) for p in prows]
        maps = [{int(pos): i for i, pos in enumerate(p["seq_positions"])} for p in prows]

        enc = [tok(p, add_special_tokens=False).input_ids for p in prefixes]
        plen = len(enc[0])
        assert all(len(e) == plen for e in enc), f"{entry}: prefixes differ in length"
        max_new = min(a.max_model_len - plen, a.contact_mult * L + 128)
        bs = a.batch_size or batch_size_for(plen + max_new)

        rows = []
        total_gen = 0
        t0 = time.time()
        torch.manual_seed(a.seed)
        for s in range(0, len(enc), bs):
            ids = torch.tensor(enc[s:s + bs], device="cuda")
            attn = torch.ones_like(ids)
            with torch.no_grad():
                out = model.generate(
                    ids, attention_mask=attn,
                    do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                    max_new_tokens=max_new, eos_token_id=end_id, pad_token_id=end_id,
                )
            seqs = out[:, plen:]  # [B, gen_len]
            for bi in range(seqs.shape[0]):
                gi = s + bi
                gen_ids = seqs[bi].tolist()
                if end_id in gen_ids:
                    cut = gen_ids.index(end_id) + 1
                    finished = True
                else:
                    cut = len(gen_ids)
                    finished = False
                gen_ids = gen_ids[:cut]
                ntok = len(gen_ids)
                total_gen += ntok
                n_contact_stmts = sum(1 for g in gen_ids if g == contact_id)

                text = tok.decode(gen_ids, skip_special_tokens=False)
                pred = parse_pred(text, maps[gi])
                sc = score_rollout(pred, gtb)
                rows.append(dict(
                    r=rkeys[gi], n_gen_tokens=ntok, finished=finished,
                    n_contact_stmts=n_contact_stmts, n_pred=len(pred), **sc,
                ))
            del out, seqs
        gen_s = time.time() - t0

        pq.write_table(pa.Table.from_pylist(rows),
                       f"{a.out}/rollout_metrics/{entry}.parquet")

        tps = total_gen / gen_s if gen_s > 0 else 0.0
        mean_pred = sum(r["n_pred"] for r in rows) / len(rows)
        mean_tok = total_gen / len(rows)
        frac_fin = sum(r["finished"] for r in rows) / len(rows)
        mean_rec = sum(r["all_rec"] for r in rows) / len(rows)
        best_f1 = max(r["all_f1"] for r in rows)
        trow = dict(entry_id=entry, dataset=t["dataset"], stem=t["stem"], L=L,
                    n_gt=n_gt, n_gt_all=int(t["n_gt_all"]), plen=plen, max_new=max_new,
                    batch_size=bs, n_rollouts=len(rows),
                    mean_gen_tokens=round(mean_tok, 1), frac_finished=round(frac_fin, 3),
                    mean_n_pred=round(mean_pred, 1),
                    mean_pred_over_gt=round(mean_pred / n_gt, 3) if n_gt else float("nan"),
                    mean_all_rec=round(mean_rec, 3), best_all_f1=round(best_f1, 3),
                    tokens_per_s=round(tps, 1), gen_seconds=round(gen_s, 1))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(trow.keys()))
        w.writeheader(); w.writerow(trow)
        with open(f"{a.out}/timings/{entry}.csv", "w") as fh:
            fh.write(buf.getvalue())

        print(f"  [{n+1}/{len(todo)}] {entry} L={L} n_gt={n_gt}  {gen_s:.1f}s {tps:.0f}tok/s"
              f"  mean_pred={mean_pred:.1f} (pred/gt={mean_pred/n_gt:.2f})"
              f"  mean_tok={mean_tok:.0f}  finished={frac_fin:.3f}"
              f"  mean_rec={mean_rec:.3f} best_f1={best_f1:.3f}", flush=True)

    print(f"\nDONE: {len(todo)} targets, wall {time.time()-t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
