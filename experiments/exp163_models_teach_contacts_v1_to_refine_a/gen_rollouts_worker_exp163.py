# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM/TPU rollout worker (exp163 — rollout-refinement). For a shard of the
targets, generate ``--n-rollouts`` contacts-v1 rollouts per target, score every
rollout's precision/recall/F1 (per separation band), and write the parsed
predictions to **per-shard aggregate parquets** that the refinement-corpus
builder consumes.

Forked from exp98's ``gen_rollouts_worker_vllm_tpu.py`` with the changes needed to
scale to ~1M proteins (issue #163, SCALE_PLAN.md §A):

  * defaults tuned for exp163: ``--n-rollouts 24``, ``--top-k -1`` (disable top-k;
    vLLM convention — the #142 under-generation fix), ``--tensor-parallel-size 4``
    (shard across all four v5p-8 chips);
  * ``logprobs`` dropped from sampling (exp163 uses only ``pred``; logprobs make
    generation ~3.7x slower), so the per-rollout ``nll`` / ``nll_per_tok`` fields
    are gone too;
  * **output rewritten for scale.** exp98 wrote 3 tiny files *per target*
    (``rollout_metrics/<entry>.parquet`` + ``best_rollouts/<entry>.json`` +
    ``timings/<entry>.csv``) → ~3M GCS objects at 1M proteins, which melts
    publishing. Instead we accumulate rollout rows in memory (every row tagged
    with its ``entry_id``) and flush a PART file every ``FLUSH_EVERY`` targets:
        rollout_metrics/shard-<i>-part-<k>.parquet   (<i> = this shard's index)
        timings/shard-<i>.csv                        (aggregate; rewritten each flush)
    ``best_rollouts`` is dropped entirely.

Same two recipes (``--mode``) as exp98:
  * ``resample`` (default, exp82's settled recipe) — each rollout uses a fresh
    document realization: N distinct prefixes, one sample each.
  * ``nsample`` — one fixed realization, ``n=n_rollouts`` samples (shared-prefix
    KV cache; the throughput-comparison baseline).

Per-rollout row: ``entry_id, r, n_gen_tokens, finished, n_pred, pred`` (flattened
seq-index ``[i0,j0,…]``) + the per-band metrics from ``score_rollout``
(``{all,short,med,long}_{npred,tp,prec,rec,f1}``).

Marinfold-free: reads pre-built resampled prefixes from ``--prompts``
(gen_prompts.py) and ground truth from ``--targets`` — runs in the marin
checkout's vllm/tpu env (mirrors exp89's thin worker). Re-runs / re-shards are
resume-safe: existing ``shard-<i>-part-*.parquet`` are read on startup and their
``entry_id``s skipped.

    # local / direct (one shard):
    python gen_rollouts_worker_exp163.py \
        --model gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf_bf16/step-35679 \
        --targets gs://.../exp163_.../targets.parquet \
        --prompts gs://.../exp163_.../prompts \
        --out     gs://.../exp163_.../runs/full \
        --shard 0/16 --n-rollouts 24 --top-k -1 --tensor-parallel-size 4

    # iris (v5p-8, us-east5); fan out over --shard i/N, one job per shard i:
    WANDB_API_KEY=... uv run iris --cluster marin job run \
        --tpu=v5p-8 --extra=vllm --extra=tpu --zone=us-east5-a \
        -- python experiments/exp163_models_teach_contacts_v1_to_refine_a/gen_rollouts_worker_exp163.py \
        --model gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf_bf16/step-35679 \
        --targets gs://.../exp163_.../targets.parquet \
        --prompts gs://.../exp163_.../prompts \
        --out     gs://.../exp163_.../runs/full \
        --shard 0/16 --n-rollouts 24 --top-k -1 --tensor-parallel-size 4
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from rollout_metrics import BANDS, gt_by_band, parse_pred, parse_sections, score_rollout

# Accumulate rollout rows across this many targets before flushing a part file.
FLUSH_EVERY = 2000
_PART_RE = re.compile(r"-part-(\d+)\.parquet$")


def stage(model_path):
    """Download a remote (gs://|s3://) model dir to local tmp (vLLM needs local files)."""
    if "://" not in model_path:  # local path -> pass through; gs:// or s3:// -> stage
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


def _part_index(path):
    m = _PART_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True, help="prompts dir (per-target parquet)")
    ap.add_argument("--out", required=True, help="output run dir (gs:// or local)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-rollouts", type=int, default=24)
    ap.add_argument("--mode", choices=["resample", "nsample"], default="resample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1,
                    help="-1 disables top-k (vLLM convention); the #142 under-generation fix")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=4,
                    help="shard the model across this many TPU chips (v5p-8 has 4)")
    ap.add_argument("--tpu-type", default="v5p-8")
    ap.add_argument("--limit", type=int, default=None, help="first N targets of the shard")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess all assigned targets; drops this shard's existing "
                         "part files + timings first (a clean redo)")
    ap.add_argument("--format", choices=["candidate", "multi-draft"], default="candidate",
                    help="multi-draft: the model emits SEVERAL <begin_statements> "
                         "sections in one completion, the last closed by <end>. Scores "
                         "each section separately; <end> remains the stop token.")
    ap.add_argument("--max-sections", type=int, default=8,
                    help="multi-draft: token budget is sized for this many sections")
    ap.add_argument("--save-texts", action="store_true",
                    help="also dump every rollout's text for the first target (debug)")
    a = ap.parse_args()
    si, sm = (int(x) for x in a.shard.split("/"))
    out = a.out.rstrip("/")
    ofs, _ = fsspec.core.url_to_fs(out)
    metrics_dir = f"{out}/rollout_metrics"
    timings_csv = f"{out}/timings/shard-{si}.csv"

    from vllm import LLM, SamplingParams, TokensPrompt

    with fsspec.open(a.targets, "rb") as fh:
        targets = {t["entry_id"]: t for t in pq.read_table(fh).to_pylist()}
    stems = sorted(targets)
    mine = stems[si::sm]
    if a.limit:
        mine = mine[: a.limit]

    # Resume: this shard's already-flushed part files carry the done entry_ids.
    # Read them, skip those targets, and continue part numbering after the last
    # part so re-runs / re-shards never clobber prior output.
    def _existing_parts():
        try:
            return sorted(ofs.glob(f"{metrics_dir}/shard-{si}-part-*.parquet"))
        except FileNotFoundError:
            return []

    parts = _existing_parts()
    if a.overwrite and parts:
        for p in parts:
            ofs.rm(p)
        if ofs.exists(timings_csv):
            ofs.rm(timings_csv)
        parts = []
    part_k = max((_part_index(p) for p in parts), default=-1) + 1

    done = set()
    for p in parts:
        with ofs.open(p, "rb") as fh:
            done.update(pq.read_table(fh, columns=["entry_id"]).column("entry_id").to_pylist())

    # Prior timings (crash-safety across resumes): keep the rows for already-done
    # targets verbatim so the aggregate CSV stays cumulative. These are re-read as
    # strings and only ever re-serialized, so we never do arithmetic on them.
    prior_timings = []
    if done and ofs.exists(timings_csv):
        with ofs.open(timings_csv, "r") as fh:
            prior_timings = [r for r in csv.DictReader(fh) if r.get("entry_id") in done]

    if not a.overwrite:
        todo = [e for e in mine if e not in done]
        print(f"shard {si}/{sm}: {len(mine)} assigned, {len(mine)-len(todo)} already done, "
              f"{len(todo)} to do (resuming at part {part_k})", flush=True)
        mine = todo
    print(f"shard {si}/{sm}: {len(mine)}/{len(stems)} targets, mode={a.mode}, "
          f"n_rollouts={a.n_rollouts}, top_k={a.top_k}, tp={a.tensor_parallel_size}, "
          f"tpu={a.tpu_type}", flush=True)
    if not mine:
        print(f"SHARD_DONE {si}/{sm}: nothing to do", flush=True)
        return

    t_load = time.time()
    llm = LLM(model=stage(a.model), max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size, enforce_eager=True, dtype="bfloat16")
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids("<end>")
    # v2 multi-draft keeps <end> as the document terminator (drafts are closed by
    # the next <begin_statements>, not by <end>), so the stop token is unchanged.
    # Only the token BUDGET grows, to leave room for several sections.
    stop_id = end_id
    sections_budget = a.max_sections if a.format == "multi-draft" else 1
    startup_s = time.time() - t_load
    print(f"model loaded in {startup_s:.0f}s (end_id={end_id})", flush=True)

    buf = []          # rollout rows accumulated since the last flush
    timings = []      # this run's per-target timing rows (typed; used for summary)

    def write_timings():
        rows = prior_timings + timings
        if not rows:
            return
        tbuf = io.StringIO()
        tw = csv.DictWriter(tbuf, fieldnames=list(rows[0].keys()))
        tw.writeheader()
        tw.writerows(rows)
        with fsspec.open(timings_csv, "w") as fh:
            fh.write(tbuf.getvalue())

    def flush_part():
        nonlocal part_k, buf
        if not buf:
            return
        dest = f"{metrics_dir}/shard-{si}-part-{part_k}.parquet"
        write_parquet(pa.Table.from_pylist(buf), dest)
        print(f"  flushed {len(buf)} rows -> {os.path.basename(dest)}", flush=True)
        part_k += 1
        buf = []

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
            max_new = min(a.max_model_len - max(len(x) for x in ids_list),
                          (4 * L + 64) * sections_budget)
            sp = SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p,
                                top_k=a.top_k, max_tokens=max_new, stop_token_ids=[stop_id])
            t0 = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids_list],
                                sp, use_tqdm=False)
            gen_s = time.time() - t0
            completions = [(o.outputs[0], maps[i], rkeys[i], prefixes[i]) for i, o in enumerate(outs)]
        else:  # nsample
            p0 = prows[0]
            ids = tok(p0["prefix"], add_special_tokens=False).input_ids
            m0 = {int(pos): i for i, pos in enumerate(p0["seq_positions"])}
            max_new = min(a.max_model_len - len(ids), (4 * L + 64) * sections_budget)
            sp = SamplingParams(n=a.n_rollouts, temperature=a.temperature, top_p=a.top_p,
                                top_k=a.top_k, max_tokens=max_new, stop_token_ids=[stop_id])
            t0 = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            gen_s = time.time() - t0
            completions = [(co, m0, j, p0["prefix"]) for j, co in enumerate(outs[0].outputs)]

        rows = []
        total_gen = 0
        all_texts = [] if (a.save_texts and n == 0) else None
        for co, m, rkey, prefix in completions:
            tok_ids = co.token_ids
            ntok = len(tok_ids)
            total_gen += ntok
            text = tok.decode(tok_ids, skip_special_tokens=False)
            extra: dict = {}
            if a.format == "multi-draft":
                secs = parse_sections(text, m)
                f1s = [score_rollout(p, gtb)["all_f1"] for p in secs]
                pred = secs[-1]                      # the LAST section is the answer
                # Diagnostics for the premise the RL plan rests on: do successive
                # drafts actually improve, and are they even different?
                jac = [len(x & y) / max(1, len(x | y)) for x, y in zip(secs, secs[1:])]
                extra = dict(
                    n_sections=len(secs),
                    section_f1=[float(v) for v in f1s],
                    best_f1=float(max(f1s)) if f1s else float("nan"),
                    last_f1=float(f1s[-1]) if f1s else float("nan"),
                    first_f1=float(f1s[0]) if f1s else float("nan"),
                    # fraction of consecutive steps that improve
                    frac_improving=(float(np.mean([b > a_ for a_, b in zip(f1s, f1s[1:])]))
                                    if len(f1s) > 1 else float("nan")),
                    # 1.0 = draft t+1 identical to draft t (the copy failure mode)
                    mean_jaccard=float(np.mean(jac)) if jac else float("nan"),
                )
            else:
                pred = parse_pred(text, m)
            sc = score_rollout(pred, gtb)
            finished = co.finish_reason == "stop"
            # flattened predicted contacts [i0,j0,i1,j1,…] (seq-index, sep>=6,
            # deduped) — the corpus builder reshapes this to (-1, 2).
            pred_flat = [int(x) for pr in sorted(pred) for x in pr]
            rows.append(dict(entry_id=entry, r=int(rkey), n_gen_tokens=ntok,
                              finished=finished, n_pred=len(pred), pred=pred_flat,
                              **sc, **extra))
            if all_texts is not None:
                all_texts.append(dict(r=int(rkey), text=text))

        buf.extend(rows)
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
        print(f"  [{n+1}/{len(mine)}] {entry} L={L} n_gt={len(gt)}  "
              f"{gen_s:.1f}s  {tps:.0f} tok/s  "
              f"mean_rec={sum(r['all_rec'] for r in rows)/len(rows):.3f}  "
              f"frac_fin={trow['frac_finished']:.2f}", flush=True)

        # Flush a part file every FLUSH_EVERY targets; rewrite the aggregate
        # timings CSV alongside it for crash-safety.
        if (n + 1) % FLUSH_EVERY == 0:
            flush_part()
            write_timings()

    flush_part()          # remainder
    write_timings()

    total_tok = sum(t["total_gen_tokens"] for t in timings)
    total_gen_s = sum(t["gen_seconds"] for t in timings)
    wall = time.time() - t_all
    rate = total_tok / total_gen_s if total_gen_s > 0 else 0.0
    print(f"\nSHARD_DONE {si}/{sm}: {len(timings)} targets, {total_tok} gen tokens, "
          f"gen {total_gen_s:.0f}s ({rate:.0f} tok/s), "
          f"wall {wall:.0f}s (+{startup_s:.0f}s startup)", flush=True)


if __name__ == "__main__":
    main()
