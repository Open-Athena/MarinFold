"""vLLM/TPU only-correct rollout worker (exp100). For a shard of the targets,
generate ``--n-rollouts`` *constrained* contacts-v1 rollouts per target — at every
step the model may only emit a token that completes a **true, not-yet-emitted**
contact, and ``<end>`` is masked until all true contacts are out — then score each
rollout's **unmodified** (full-vocabulary) NLL with a separate ``prompt_logprobs``
pass and keep the most-likely one (lowest structure-section NLL) as the regenerated
training document for that protein (issue #100).

Marinfold-free: reads pre-built resampled prefixes from ``--prompts`` (gen_prompts.py)
and ground-truth contacts from ``--targets``, so it runs in the marin checkout's
vllm/tpu env on iris (mirrors exp89/exp98's thin worker). The constraint grammar
lives in ``constrained_grammar.py`` (pure-Python, unit-tested) and is the only new
piece vs exp98.

Two passes per target, on one shared LLM:
  A) constrained generate  — per-rollout SamplingParams, each carrying its own
     ContactConstraint logits-processor (forces the only-correct grammar).
  B) prompt_logprobs score — re-score ``prefix + generated`` under the *unmodified*
     model (exp89's proven NLL path) to rank rollouts by the real likelihood, not
     the masked one we sampled from.

Writes to ``--out`` (GCS):
  nll/<entry_id>.parquet        one row per rollout (structure/doc NLL, timing, correctness check)
  documents/<entry_id>.json     the selected (lowest structure-NLL) regenerated document
  all_documents/<entry_id>.json all N rollout documents verbatim (token order preserved)
  timings/<entry_id>.csv        per-target timing

  python gen_constrained_worker_vllm_tpu.py \
      --model gs://marin-us-east5/checkpoints/prot-exp75-...-bc3084/hf_bf16/step-35679 \
      --targets gs://.../exp100_only_correct_contacts_v1_train/targets.parquet \
      --prompts gs://.../exp100_only_correct_contacts_v1_train/prompts \
      --out     gs://.../exp100_only_correct_contacts_v1_train/runs/calib \
      --shard 0/1 --n-rollouts 10 --tpu-type v5p-8 --tensor-parallel-size 4
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from constrained_grammar import ContactConstraint
from rollout_metrics import gt_by_band, parse_pred, score_rollout


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
    ap.add_argument("--n-rollouts", type=int, default=10)
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

    def done(entry):
        return (ofs.exists(f"{out}/nll/{entry}.parquet")
                and ofs.exists(f"{out}/documents/{entry}.json"))

    if not a.overwrite:
        todo = [e for e in mine if not done(e)]
        print(f"shard {si}/{sm}: {len(mine)} assigned, {len(mine)-len(todo)} already done, "
              f"{len(todo)} to do", flush=True)
        mine = todo
    print(f"shard {si}/{sm}: {len(mine)}/{len(stems)} targets, n_rollouts={a.n_rollouts}, "
          f"tp={a.tensor_parallel_size}, tpu={a.tpu_type}", flush=True)
    if not mine:
        print(f"SHARD_DONE {si}/{sm}: nothing to do", flush=True)
        return

    t_load = time.time()
    llm = LLM(model=stage(a.model), max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size, enforce_eager=True,
              dtype="bfloat16")
    tok = llm.get_tokenizer()
    contact_id = tok.convert_tokens_to_ids("<contact>")
    end_id = tok.convert_tokens_to_ids("<end>")
    startup_s = time.time() - t_load
    print(f"model loaded in {startup_s:.0f}s (contact_id={contact_id}, end_id={end_id})", flush=True)

    def pos_token_id(pos):
        return tok.convert_tokens_to_ids(f"<p{pos}>")

    timings = []
    t_all = time.time()
    for n, entry in enumerate(mine):
        t = targets[entry]
        L = int(t["L"])
        gt_seq = {(int(i), int(j)) for i, j in t["gt_contacts"]}
        gtb = gt_by_band(gt_seq)

        with fsspec.open(f"{a.prompts}/{entry}.parquet", "rb") as fh:
            prows = pq.read_table(fh).to_pylist()[: a.n_rollouts]

        # --- Constrained generation with folded-in unmodified-NLL capture ---
        # (one SamplingParams + one ContactConstraint per rollout; the processor
        # records each realized token's pre-mask full-vocab logprob, so there is
        # no separate prompt_logprobs pass — which returns None on TPU anyway.)
        prompts, sps, cons, meta = [], [], [], []
        for p in prows:
            prompt_ids = tok(p["prefix"], add_special_tokens=False).input_ids
            seq_positions = list(p["seq_positions"])         # seq index -> position
            pos_to_seq = {int(pos): i for i, pos in enumerate(seq_positions)}
            # GT contacts (seq-index space) -> position-token-id pairs for the mask
            gt_pos_ids = [(pos_token_id(seq_positions[i]), pos_token_id(seq_positions[j]))
                          for i, j in gt_seq]
            con = ContactConstraint(prompt_ids, gt_pos_ids,
                                    contact_id=contact_id, end_id=end_id)
            prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
            sps.append(SamplingParams(
                n=1, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                max_tokens=con.max_new_tokens() + 4, stop_token_ids=[end_id],
                logits_processors=[con]))
            cons.append(con)
            meta.append(dict(r=int(p["r"]), prefix=p["prefix"], pos_to_seq=pos_to_seq))
        t0 = time.time()
        gouts = llm.generate(prompts, sps, use_tqdm=False)
        gen_s = time.time() - t0

        gen_ids = [list(o.outputs[0].token_ids) for o in gouts]
        gen_text = [tok.decode(g, skip_special_tokens=False) for g in gen_ids]

        rows, docs, total_gen = [], [], 0
        for i in range(len(prows)):
            g = gen_ids[i]
            total_gen += len(g)
            cons[i].finalize(g)
            struct_nll = cons[i].struct_nll()
            nstruct = len(cons[i].token_logprobs)
            # correctness check: the constrained doc must be 100% correct + full recall
            pred = parse_pred(gen_text[i], meta[i]["pos_to_seq"])
            sc = score_rollout(pred, gtb)
            finished = gouts[i].outputs[0].finish_reason == "stop"
            # count statements by <contact> tokens — robust to whether vLLM
            # includes the <end> stop token in the output ids.
            n_contacts = sum(1 for t in g if t == contact_id)
            rows.append(dict(
                r=meta[i]["r"], n_gen_tokens=len(g), n_contacts=n_contacts,
                finished=finished, n_pred=len(pred),
                all_prec=sc["all_prec"], all_rec=sc["all_rec"], all_f1=sc["all_f1"],
                struct_nll=struct_nll, struct_ntok=nstruct,
                struct_nll_per_tok=(struct_nll / nstruct) if nstruct else math.nan))
            docs.append(dict(r=meta[i]["r"], document=meta[i]["prefix"] + " " + gen_text[i],
                             struct_nll=struct_nll,
                             n_contacts=n_contacts, finished=finished,
                             all_prec=sc["all_prec"], all_rec=sc["all_rec"],
                             pred_contacts=sorted([list(p) for p in pred])))

        # selection: lowest structure-section NLL (the model's preferred ordering)
        best = min(range(len(rows)), key=lambda i: rows[i]["struct_nll"])
        n_correct = sum(1 for r in rows if r["all_prec"] == 1.0 and r["all_rec"] == 1.0)

        write_parquet(pa.Table.from_pylist(rows), f"{out}/nll/{entry}.parquet")
        sel = docs[best]
        sel_meta = dict(entry_id=entry, L=L, n_gt=len(gt_seq),
                        n_rollouts=len(rows), n_correct=n_correct,
                        selected_by="struct_nll", selected=sel,
                        sampling=dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k))
        with fsspec.open(f"{out}/documents/{entry}.json", "w") as fh:
            json.dump(sel_meta, fh)
        with fsspec.open(f"{out}/all_documents/{entry}.json", "w") as fh:
            json.dump(dict(entry_id=entry, documents=docs), fh)

        tps = total_gen / gen_s if gen_s > 0 else 0.0
        trow = dict(entry_id=entry, L=L, n_gt=len(gt_seq), n_rollouts=len(rows),
                    n_correct=n_correct, tensor_parallel=a.tensor_parallel_size,
                    total_gen_tokens=total_gen, gen_seconds=round(gen_s, 3),
                    tokens_per_s=round(tps, 1),
                    mean_gen_tokens=round(total_gen / len(rows), 1),
                    best_struct_nll=round(rows[best]["struct_nll"], 3),
                    tpu_type=a.tpu_type)
        timings.append(trow)
        tbuf = io.StringIO()
        tw = csv.DictWriter(tbuf, fieldnames=list(trow.keys()))
        tw.writeheader(); tw.writerow(trow)
        with fsspec.open(f"{out}/timings/{entry}.csv", "w") as fh:
            fh.write(tbuf.getvalue())
        print(f"  [{n+1}/{len(mine)}] {entry} L={L} n_gt={len(gt_seq)}  "
              f"gen {gen_s:.1f}s  {tps:.0f} tok/s  "
              f"correct={n_correct}/{len(rows)}  best_struct_nll={rows[best]['struct_nll']:.1f}",
              flush=True)
        if n_correct != len(rows):
            print(f"  WARNING {entry}: {len(rows)-n_correct} rollouts not 100%-correct "
                  f"(constraint or mapping bug)", flush=True)

    total_tok = sum(t["total_gen_tokens"] for t in timings)
    total_gen_s = sum(t["gen_seconds"] for t in timings)
    wall = time.time() - t_all
    print(f"\nSHARD_DONE {si}/{sm}: {len(timings)} targets, {total_tok} gen tokens, "
          f"gen {total_gen_s:.0f}s ({total_tok/total_gen_s:.0f} tok/s), "
          f"wall {wall:.0f}s (+{startup_s:.0f}s startup)", flush=True)


if __name__ == "__main__":
    main()
