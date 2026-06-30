"""vLLM/TPU rollout worker (exp98). For a shard of the exp98 targets, generate
``--n-rollouts`` contacts-v1 rollouts per target, score every rollout's
precision/recall/F1 (per separation band), and save the best-recall + best-F1
rollouts verbatim. Marinfold-free: reads pre-built resampled prefixes from
``--prompts`` (see gen_prompts.py) and ground truth from ``--targets`` — runs in
the marin checkout's vllm/tpu env (mirrors exp89's thin worker).

Two recipes (``--mode``):
  * ``resample`` (default, exp82's settled recipe) — each rollout uses a fresh
    document realization: 1000 distinct prefixes, one sample each.
  * ``nsample`` — one fixed realization, ``n=n_rollouts`` samples (shared-prefix
    KV cache; the throughput-comparison baseline).

Writes to ``--out`` (GCS):
  rollout_metrics/<entry_id>.parquet   one row per rollout (per-band metrics)
  best_rollouts/<entry_id>.json        best-recall + best-F1 rollouts (full doc)
  timings/<shard>.csv                  per-target generation timing

  python gen_rollouts_worker_vllm_tpu.py \
      --model gs://marin-us-east5/checkpoints/prot-exp75-...-bc3084/hf_bf16/step-35679 \
      --targets gs://.../exp98_rollouts_contacts_v1_train/targets.parquet \
      --prompts gs://.../exp98_rollouts_contacts_v1_train/prompts \
      --out     gs://.../exp98_rollouts_contacts_v1_train/runs/calib \
      --shard 0/1 --n-rollouts 1000 --mode resample --tpu-type v5p-8
"""
import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import time

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from rollout_metrics import BANDS, gt_by_band, parse_pred, score_rollout


def stage(model_path):
    """Download a gs:// model dir to local tmp (TPU worker needs local files)."""
    if not model_path.startswith("gs://"):
        return model_path
    fs, root = url_to_fs(model_path.rstrip("/"))
    local = os.path.join(tempfile.gettempdir(), "cv1model",
                         hashlib.sha256(model_path.encode()).hexdigest()[:12])
    os.makedirs(local, exist_ok=True)
    for e in fs.find(root, detail=True, maxdepth=1).values():
        if e.get("type") != "file":
            continue
        name = os.path.basename(e["name"])
        lp = os.path.join(local, name)
        if not (os.path.exists(lp) and os.path.getsize(lp) == e.get("size")):
            fs.get(e["name"], lp)
    return local


def write_parquet(table, dest):
    with fsspec.open(dest, "wb") as fh:
        pq.write_table(table, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True, help="prompts dir (per-target parquet)")
    ap.add_argument("--out", required=True, help="output run dir (gs:// or local)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-rollouts", type=int, default=1000)
    ap.add_argument("--mode", choices=["resample", "nsample"], default="resample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="shard the model across this many TPU chips (v5p-8 has 4)")
    ap.add_argument("--tpu-type", default="v5p-8")
    ap.add_argument("--limit", type=int, default=None, help="first N targets of the shard")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run targets even if their outputs already exist")
    ap.add_argument("--save-texts", action="store_true",
                    help="also dump every rollout's text for the first target (debug)")
    a = ap.parse_args()
    si, sm = (int(x) for x in a.shard.split("/"))
    out = a.out.rstrip("/")
    ofs, _ = fsspec.core.url_to_fs(out)

    from vllm import LLM, SamplingParams, TokensPrompt

    with fsspec.open(a.targets, "rb") as fh:
        targets = {t["entry_id"]: t for t in pq.read_table(fh).to_pylist()}
    stems = sorted(targets)
    mine = stems[si::sm]
    if a.limit:
        mine = mine[: a.limit]

    # Resume: skip targets whose outputs are already complete (survives engine
    # restarts / shard re-runs). Both the metrics parquet and the best-rollouts
    # JSON must exist.
    def done(entry):
        return (ofs.exists(f"{out}/rollout_metrics/{entry}.parquet")
                and ofs.exists(f"{out}/best_rollouts/{entry}.json"))

    if not a.overwrite:
        todo = [e for e in mine if not done(e)]
        print(f"shard {si}/{sm}: {len(mine)} assigned, {len(mine)-len(todo)} already done, "
              f"{len(todo)} to do", flush=True)
        mine = todo
    print(f"shard {si}/{sm}: {len(mine)}/{len(stems)} targets, mode={a.mode}, "
          f"n_rollouts={a.n_rollouts}, tp={a.tensor_parallel_size}, tpu={a.tpu_type}", flush=True)
    if not mine:
        print(f"SHARD_DONE {si}/{sm}: nothing to do", flush=True)
        return

    t_load = time.time()
    llm = LLM(model=stage(a.model), max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size, enforce_eager=True, dtype="bfloat16")
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids("<end>")
    startup_s = time.time() - t_load
    print(f"model loaded in {startup_s:.0f}s (end_id={end_id})", flush=True)

    band_cols = [f"{b}_{m}" for b in BANDS for m in ("npred", "tp", "prec", "rec", "f1")]
    timings = []
    t_all = time.time()
    for n, entry in enumerate(mine):
        t = targets[entry]
        L = int(t["L"])
        gt = {(int(i), int(j)) for i, j in t["gt_contacts"]}
        gtb = gt_by_band(gt)

        with fsspec.open(f"{a.prompts}/{entry}.parquet", "rb") as fh:
            prows = pq.read_table(fh).to_pylist()
        prows = prows[: a.n_rollouts]

        if a.mode == "resample":
            ids_list = [tok(p["prefix"], add_special_tokens=False).input_ids for p in prows]
            maps = [{int(pos): i for i, pos in enumerate(p["seq_positions"])} for p in prows]
            rkeys = [p["r"] for p in prows]
            prefixes = [p["prefix"] for p in prows]
            max_new = min(a.max_model_len - max(len(x) for x in ids_list), 4 * L + 64)
            sp = SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p,
                                top_k=a.top_k, max_tokens=max_new, stop_token_ids=[end_id])
            t0 = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids_list],
                                sp, use_tqdm=False)
            gen_s = time.time() - t0
            completions = [(o.outputs[0], maps[i], rkeys[i], prefixes[i]) for i, o in enumerate(outs)]
        else:  # nsample
            p0 = prows[0]
            ids = tok(p0["prefix"], add_special_tokens=False).input_ids
            m0 = {int(pos): i for i, pos in enumerate(p0["seq_positions"])}
            max_new = min(a.max_model_len - len(ids), 4 * L + 64)
            sp = SamplingParams(n=a.n_rollouts, temperature=a.temperature, top_p=a.top_p,
                                top_k=a.top_k, max_tokens=max_new, stop_token_ids=[end_id])
            t0 = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            gen_s = time.time() - t0
            completions = [(co, m0, j, p0["prefix"]) for j, co in enumerate(outs[0].outputs)]

        rows = []
        total_gen = 0
        best_rec = best_f1 = None
        all_texts = [] if (a.save_texts and n == 0) else None
        for co, m, rkey, prefix in completions:
            tok_ids = co.token_ids
            total_gen += len(tok_ids)
            text = tok.decode(tok_ids, skip_special_tokens=False)
            pred = parse_pred(text, m)
            sc = score_rollout(pred, gtb)
            finished = co.finish_reason == "stop"
            rows.append(dict(r=int(rkey), n_gen_tokens=len(tok_ids), finished=finished,
                             n_pred=len(pred), **sc))
            if all_texts is not None:
                all_texts.append(dict(r=int(rkey), text=text))
            rec, f1, prec = sc["all_rec"], sc["all_f1"], sc["all_prec"]
            cand = dict(r=int(rkey), precision=prec, recall=rec, f1=f1,
                        n_gen_tokens=len(tok_ids), finished=finished,
                        document=prefix + " " + text,
                        pred_contacts=sorted([list(p) for p in pred]))
            if best_rec is None or (rec, f1) > (best_rec["recall"], best_rec["f1"]):
                best_rec = cand
            if best_f1 is None or (f1, rec) > (best_f1["f1"], best_f1["recall"]):
                best_f1 = cand

        write_parquet(pa.Table.from_pylist(rows), f"{out}/rollout_metrics/{entry}.parquet")
        meta = dict(entry_id=entry, L=L, n_gt=len(gt),
                    gt_by_band={b: len(gtb[b]) for b in BANDS},
                    mode=a.mode, n_rollouts=len(rows),
                    sampling=dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k),
                    best_recall=best_rec, best_f1=best_f1)
        with fsspec.open(f"{out}/best_rollouts/{entry}.json", "w") as fh:
            json.dump(meta, fh)
        if all_texts is not None:
            with fsspec.open(f"{out}/debug_texts/{entry}.json", "w") as fh:
                json.dump(all_texts, fh)

        tps = total_gen / gen_s if gen_s > 0 else 0.0
        trow = dict(entry_id=entry, L=L, n_gt=len(gt), n_rollouts=len(rows),
                    mode=a.mode, tensor_parallel=a.tensor_parallel_size,
                    total_gen_tokens=total_gen,
                    gen_seconds=round(gen_s, 3), tokens_per_s=round(tps, 1),
                    mean_gen_tokens=round(total_gen / len(rows), 1),
                    frac_finished=round(sum(r["finished"] for r in rows) / len(rows), 3),
                    tpu_type=a.tpu_type)
        timings.append(trow)
        # per-target timing file (survives engine restarts / shard re-runs).
        tbuf = io.StringIO()
        tw = csv.DictWriter(tbuf, fieldnames=list(trow.keys()))
        tw.writeheader(); tw.writerow(trow)
        with fsspec.open(f"{out}/timings/{entry}.csv", "w") as fh:
            fh.write(tbuf.getvalue())
        print(f"  [{n+1}/{len(mine)}] {entry} L={L} n_gt={len(gt)}  "
              f"{gen_s:.1f}s  {tps:.0f} tok/s  mean_rec={sum(r['all_rec'] for r in rows)/len(rows):.3f}  "
              f"best_f1={best_f1['f1']:.3f} best_rec={best_rec['recall']:.3f}", flush=True)

    total_tok = sum(t["total_gen_tokens"] for t in timings)
    total_gen_s = sum(t["gen_seconds"] for t in timings)
    wall = time.time() - t_all
    print(f"\nSHARD_DONE {si}/{sm}: {len(timings)} targets, {total_tok} gen tokens, "
          f"gen {total_gen_s:.0f}s ({total_tok/total_gen_s:.0f} tok/s), "
          f"wall {wall:.0f}s (+{startup_s:.0f}s startup)", flush=True)


if __name__ == "__main__":
    main()
